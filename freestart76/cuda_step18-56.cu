/*****
  Copyright (C) 2015 Pierre Karpman, INRIA France/Nanyang Technological University Singapore (-2016), CWI (2016/2017), L'Universite Grenoble Alpes (2017-)
            (C) 2015 Thomas Peyrin, Nanyang Technological University Singapore
            (C) 2015 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1_gpu_nearcollisionattacks source-code and released under the MIT License
*****/

/************ TODO TODO **********\
- clean up code (enable opts, remove old code)
- test performance on older gen cards
\************ TODO TODO *********/


//// main prepocessor flags

// enables managed cyclic buffers and CPU verification of GPU results
//#define DEBUG1 

// disabling temporary buffer will force writes to directly go to main buffer
//#define DISABLE_TMP_BUF 

// enable packed SOL22 substep
#define PAC22

// enable 
#define NEW_Q56

#define BLOCKS 26
#define THREADS_PER_BLOCK 512













#include "main.hpp"
#include "cuda_cyclicbuffer.hpp"
#include "sha1detail.hpp"

#include <timer.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <iomanip>
#include <stdexcept>


using namespace hashclash;
using namespace std;

#define CUDA_ASSERT(s) 	{ cudaError_t err = s; if (err != cudaSuccess) { throw std::runtime_error("CUDA command returned : " + string(cudaGetErrorString(err)) + "!"); }  }




// set other defines based on main preprocessor flags
#ifdef DEBUG1
#ifndef USE_MANAGED
#define USE_MANAGED
#endif
#define VERIFY_GPU_RESULTS
#endif // DEBUG1

#ifdef USE_MANAGED
#define MANAGED __managed__
#else
#define MANAGED
#endif



// DEFINE CYCLIC BUFFER TYPES
// MASK VERSION
// template<size_t N> struct cyclic_buffer_control_mask_t;
// template<size_t N> struct cyclic_buffer_control_mask_readlock_t;
// template<size_t N, typename val_t = uint32_t, size_t val_cnt = 1, typename control_type = cyclic_buffer_control_mask_t<N> > struct cyclic_buffer_mask_t;
// CAS VERSION
// template<size_t N> struct cyclic_buffer_control_cas_t;
// template<size_t N, typename val_t = uint32_t, size_t val_cnt = 1, typename control_type = cyclic_buffer_control_cas_t<N> > struct cyclic_buffer_cas_t;

// definition of cyclic buffer for 2^20 2-word elems
typedef cyclic_buffer_mask_t< (1<<20), uint32_t, 2, cyclic_buffer_control_mask_t< (1<<20) >, 1 > buffer_20_2_t; // used for block-specific buffers: fencetype = block-wide
typedef buffer_20_2_t::control_t control_20_2_t;

// definition of cyclic buffer for 2^20 22-word elems: basesol: Q12,..,Q17,m6,...,m21 [uses CAS, as it's only written by the host]
typedef cyclic_buffer_cas_t< BASESOLCOUNT, uint32_t, 22, cyclic_buffer_control_cas_t< BASESOLCOUNT > > buffer_basesol_t;
typedef buffer_basesol_t::control_t control_basesol_t;

// definition of cyclic buffer for 2^10 21-word elems: sol: Q32,..,Q36,m20,...,m35
typedef cyclic_buffer_mask_t< (1<<10), uint32_t, 21, cyclic_buffer_control_mask_t< (1<<10) > > buffer_sol_t;
typedef buffer_sol_t::control_t control_sol_t;

// definition of cyclic buffer for 2^20 11-word elems: extbasesol: Q17,..,Q21,m14,...,m18,m20,basesolidx
typedef cyclic_buffer_mask_t< (1<<20), uint32_t, 12, cyclic_buffer_control_mask_t< (1<<20) >, 1 > buffer_extbasesol_t; // used for block-specific buffers: fencetype = block-wide
typedef buffer_extbasesol_t::control_t control_extbasesol_t;



// WARP-SPECIFIC TEMPORARY BUFFER (2x 32 2-word elems)
// Usage: (all warp-wide calls)
// WARP_TMP_BUF.reset()                         // called once on kernel entrance
// WARP_TMP_BUF.write1(bool,val1,buf,ctrl)      // write 1-word elem to tmp buf, flush to mainbuf when one half of 32 is full
// WARP_TMP_BUF.write2(bool,val1,val2,buf,ctrl) // write 2-word elem to tmp buf, flush to mainbuf when one half of 32 is full
// WARP_TMP_BUF.flush1(buf,ctrl)		// call once on function exit to move remaining 1-word elems to mainbuf
// WARP_TMP_BUF.flush2(buf,ctrl)		// call once on function exit to move remaining 2-word elems to mainbuf
//
#ifdef DISABLE_TMP_BUF
class dummy_tmp_buf {
public:
	inline __device__ void reset() {}

	template<typename buffer_t, typename control_t>
	inline __device__ void flush1(buffer_t& buf, control_t& ctrl) {}

	template<typename buffer_t, typename control_t>
	inline __device__ void flush2(buffer_t& buf, control_t& ctrl) {}

	template<typename buffer_t, typename control_t>
	inline __device__ void write1(bool dowrite, uint32_t val1, buffer_t& buf, control_t& ctrl)
	{
		buf.write(ctrl, dowrite, val1);
	}

	template<typename buffer_t, typename control_t>
	inline __device__ void write2(bool dowrite, uint32_t val1, uint32_t val2, buffer_t& buf, control_t& ctrl)
	{
		buf.write(ctrl, dowrite, val1, val2);
	}
};
__shared__ dummy_tmp_buf dummytmpbuf;
#define WARP_TMP_BUF dummytmpbuf
#else
__shared__ warp_tmp_buf_t warp_tmp_buf[THREADS_PER_BLOCK/32];
#define WARP_TMP_BUF warp_tmp_buf[threadIdx.x/32]
#endif


#define MANAGED2 __managed__
//#define MANAGED2 MANAGED

/*** Main buffers declaration ***/
MANAGED __device__ buffer_basesol_t  base_solutions_buf;
__managed__ __device__ control_basesol_t base_solutions_ctl;  // always managed to detect when it's empty
#define BASESOLBUF                   base_solutions_buf
#define BASESOLCTL                   base_solutions_ctl

MANAGED __device__ buffer_20_2_t  q18_solutions_buf    [BLOCKS];
        __shared__ control_20_2_t q18_solutions_ctl;
MANAGED2 __device__ control_20_2_t q18_solutions_ctl_bu [BLOCKS];
#define Q18SOLBUF                 q18_solutions_buf    [blockIdx.x]
#define Q18SOLCTL                 q18_solutions_ctl

MANAGED __device__ buffer_20_2_t  q19_solutions_buf    [BLOCKS];
        __shared__ control_20_2_t q19_solutions_ctl;
MANAGED2 __device__ control_20_2_t q19_solutions_ctl_bu [BLOCKS];
#define Q19SOLBUF                 q19_solutions_buf    [blockIdx.x]
#define Q19SOLCTL                 q19_solutions_ctl

MANAGED __device__ buffer_20_2_t  q20_solutions_buf    [BLOCKS];
        __shared__ control_20_2_t q20_solutions_ctl;
MANAGED2 __device__ control_20_2_t q20_solutions_ctl_bu [BLOCKS];
#define Q20SOLBUF                 q20_solutions_buf    [blockIdx.x]
#define Q20SOLCTL                 q20_solutions_ctl

MANAGED __device__ buffer_20_2_t  q22_packed_solutions_buf    [BLOCKS];
        __shared__ control_20_2_t q22_packed_solutions_ctl;
MANAGED2 __device__ control_20_2_t q22_packed_solutions_ctl_bu [BLOCKS];
#define Q22PACBUF                 q22_packed_solutions_buf    [blockIdx.x]
#define Q22PACCTL                 q22_packed_solutions_ctl

MANAGED __device__ buffer_extbasesol_t  q22_solutions_buf    [BLOCKS];
        __shared__ control_extbasesol_t q22_solutions_ctl;
MANAGED2 __device__ control_extbasesol_t q22_solutions_ctl_bu [BLOCKS];
#define Q22SOLBUF                       q22_solutions_buf    [blockIdx.x]
#define Q22SOLCTL                       q22_solutions_ctl

MANAGED __device__ buffer_20_2_t  q23_solutions_buf    [BLOCKS];
        __shared__ control_20_2_t q23_solutions_ctl;
MANAGED2 __device__ control_20_2_t q23_solutions_ctl_bu [BLOCKS];
#define Q23SOLBUF                 q23_solutions_buf    [blockIdx.x]
#define Q23SOLCTL                 q23_solutions_ctl

MANAGED __device__ buffer_20_2_t  q24_solutions_buf    [BLOCKS];
        __shared__ control_20_2_t q24_solutions_ctl;
MANAGED2 __device__ control_20_2_t q24_solutions_ctl_bu [BLOCKS];
#define Q24SOLBUF                 q24_solutions_buf    [blockIdx.x]
#define Q24SOLCTL                 q24_solutions_ctl

MANAGED __device__ buffer_20_2_t  q27_solutions_buf    [BLOCKS];
        __shared__ control_20_2_t q27_solutions_ctl;
MANAGED2 __device__ control_20_2_t q27_solutions_ctl_bu [BLOCKS];
#define Q27SOLBUF                 q27_solutions_buf    [blockIdx.x]
#define Q27SOLCTL                 q27_solutions_ctl

MANAGED __device__ buffer_sol_t  q36_solutions_buf    [BLOCKS];
        __shared__ control_sol_t q36_solutions_ctl;
MANAGED2 __device__ control_sol_t q36_solutions_ctl_bu [BLOCKS];
#define Q36SOLBUF                q36_solutions_buf    [blockIdx.x]
#define Q36SOLCTL                q36_solutions_ctl

__managed__ buffer_sol_t  collision_candidates_buf;
__managed__ control_sol_t collision_candidates_ctl;
#define COLLCANDIDATEBUF  collision_candidates_buf
#define COLLCANDIDATECTL  collision_candidates_ctl






/*** performance measure stuff ***/

#ifdef USE_PERF_COUNTERS

__managed__ uint64_t main_performance_counter[BLOCKS][80];
__shared__ unsigned long long int tmp_performance_counter[80];
void performance_reset()
{
	for (unsigned b = 0; b < BLOCKS; ++b)
	{
		for (unsigned i = 0; i < 80; ++i)
		{
			main_performance_counter[b][i] = 0;
		}
	}
}
__device__ inline void performance_backup()
{
	if (threadIdx.x == 0)
	{
		for (unsigned i = 0; i < 80; ++i)
		{
			main_performance_counter[blockIdx.x][i]
				= tmp_performance_counter[i];
		}
	}
}
__device__ inline void performance_restore()
{
	if (threadIdx.x == 0)
	{
		for (unsigned i = 0; i < 80; ++i)
		{
			main_performance_counter[blockIdx.x][i]
				= tmp_performance_counter[i];
		}
	}
}
__device__ inline void performance_start_counter(unsigned i)
{
	if ((threadIdx.x&31)==0)
		atomicAdd(&tmp_performance_counter[i], -clock64());
}
__device__ inline void performance_stop_counter(unsigned i)
{
	if ((threadIdx.x&31)==0)
		atomicAdd(&tmp_performance_counter[i], clock64());
}
#define PERF_START_COUNTER(i) performance_start_counter(i);
#define PERF_STOP_COUNTER(i) performance_stop_counter(i);

void show_performance_counters()
{
	uint64_t total = 0;
	for (unsigned b = 0; b < BLOCKS; ++b)
	{
		for (unsigned i = 0; i < 80; ++i)
		{
			total += main_performance_counter[b][i];
		}
	}
	for (unsigned i = 0; i < 80; ++i)
	{
		uint64_t stepi = 0;
		for (unsigned b = 0; b < BLOCKS; ++b)
		{
			stepi += main_performance_counter[b][i];
		}
		if (stepi > 0)
		{
			cout << "Perf.counter " << i << "\t : " << setprecision(4) << (100.0*double(stepi)/double(total)) << " \t " << stepi << endl;
		}
	}
}

#else

#define PERF_START_COUNTER(i) 
#define PERF_STOP_COUNTER(i) 

#endif // USE_PERF_COUNTERS











/*** Bit condition masks for steps Q-4 to Q80, stored on the device in constant memory ***/

// QOFF: value for Q_t is at index QOFF+t in tables below
#define QOFF 4

namespace host {
#include "tables.hpp"
}

namespace dev {
#define TABLE_PREFIX __constant__
#include "tables.hpp"
}


/// Neutral bits positions //////////////
/// W14: ..................xxxxxxxxx.....
/// W15: ...............x.xxxxxxxxxx.....
/// W16: ...............x..x.xxxxxx......
/// W17: ............x.....xxxx..........
/// W18: ...............xx...............
/// W19: .................x.xxxxxxx......
/// W20: ..................xxx....x.....x
/// W21: ..............xx....x...........
#define W14NBALLM 0x00003FE0
#define W15NBALLM 0x00017FE0
#define W16NBALLM 0x00012FC0
#define W17NBALLM 0x00083C00
#define W18NBALLM 0x00018000
#define W19NBALLM 0x00005FC0
#define W20NBALLM 0x00003841
#define W21NBALLM 0x00030800

/// Per step neutral bits masks /////////
/// Q18
#define W14NBQ18M 0x00003E00 // ..................xxxxx......... (5 of 'em)
#define W15NBQ18M 0x00014000 // ...............x.x.............. (2 of 'em)
/// Q19
#define W14NBQ19M 0x000001E0 // .......................xxxx..... (4 of 'em)
#define W15NBQ19M 0x00003F00 // ..................xxxxxx........ (6 of 'em)
#define W16NBQ19M 0x00012000 // ...............x..x............. (2 of 'em)
#define W17NBQ19M 0x00080000 // ............x................... (1 of 'em)
/// Q20
#define W15NBQ20M 0x000000E0 // ........................xxx..... (3 of 'em)
#define W16NBQ20M 0x00000E00 // ....................xxx......... (3 of 'em)
/// Q21
#define W16NBQ21M 0x000001C0 // .......................xxx...... (3 of 'em)
#define W17NBQ21M 0x00003C00 // ..................xxxx.......... (4 of 'em)
#define W18NBQ21M 0x00018000 // ...............xx............... (2 of 'em)
/// Q23
#define W19NBQ23M 0x00005E00 // .................x.xxxx......... (5 of 'em)
/// Q24
#define W19NBQ24M 0x000000C0 // ........................xx...... (2 of 'em)
#define W20NBQ24M 0x00001800 // ...................xx........... (2 of 'em)
#define W21NBQ24M 0x00030000 // ..............xx................ (2 of 'em)
/// Q25
#define W19NBQ25M 0x00000100 // .......................x........ (1 of 'em)
#define W20NBQ25M 0x00002040 // ..................x......x...... (2 of 'em)
#define W21NBQ25M 0x00000800 // ....................x........... (1 of 'em)
/// Q26
#define W20NBQ26M 0x00000001 // ...............................x (1 of 'em)


/// Neutral bits packing (Q18--21) //////
/// W14 + W15 + W16 (32 bits)
/// |<13..5>||<16.....5>||<16....6>|
/// xxxxxxxxxx.xxxxxxxxxxx..x.xxxxxx
/// W17 + W18 (12 bits)
/// |<19..10>||<16,15>/base sol id |
/// x.....xxxxxx--------------------

/// Neutral bits packing (Q23--26) //////
/// W21 + W19 + W20 (30 bits + 2 padding)
/// \<17.11>||<14..6>||<13.......0>|
/// ~~xx....xx.xxxxxxxxxx....x.....x
/// 			|	ext  sol id    |
/// ~~~~~~~~~~~~--------------------

/* *** (UN)PACKING NEUTRAL BITS & BASE INDEX ****************
 * The following functions return the (un)packed neutral bits
 * with/from a proper alignement of the packed bits
 * (i.e. bit _i_ of the returned value is
 * bit _i_ of the corresponding message word)
 *
 * The packing function only return the part of the relevant
 * message word, which then needs to be ORed with the
 * rest of the packed words (always use that through the
 * step-specific functions, never directly)
 */
__device__ __host__ inline uint32_t  unpack_base_idx(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return (packed_val1 & 0xfffff);
}
__device__ __host__ inline uint32_t  unpack_ext_idx(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return (packed_val1 & 0xfffff);
}
__device__ __host__ inline uint32_t  pack_base_idx(uint32_t  base_idx)
{
	return base_idx;
}

__device__ __host__ inline uint32_t  unpack_w14_neutral_bits(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return ((packed_val0 >> 18) & (0x1ff << 5)); // 9 useful bits from 5 to 13
}
__device__ __host__ inline uint32_t  pack_w14_neutral_bits(uint32_t  w14_nb)
{
	return (w14_nb << 18);
}

__device__ __host__ inline uint32_t  unpack_w15_neutral_bits(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return ((packed_val0 >> 6) & (0xfff << 5)); // 12 (11 really) useful bits from 5 to 16
}
__device__ __host__ inline uint32_t  pack_w15_neutral_bits(uint32_t  w15_nb)
{
	return (w15_nb << 6);
}

__device__ __host__ inline uint32_t  unpack_w16_neutral_bits(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return ((packed_val0 << 6) & (0x7ff << 6)); // 11 (8 really) useful bits from 6 to 16
}
__device__ __host__ inline uint32_t  pack_w16_neutral_bits(uint32_t  w16_nb)
{
	return (w16_nb >> 6);
}

__device__ __host__ inline uint32_t  unpack_w17_neutral_bits(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return ((packed_val1 >> 12) & (0x3ff << 10)); // 10 (5 really) useful bits from 10 to 19
}
__device__ __host__ inline uint32_t  pack_w17_neutral_bits(uint32_t  w17_nb)
{
	return (w17_nb << 12);
}

__device__ __host__ inline uint32_t  unpack_w18_neutral_bits(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return ((packed_val1 >> 5) & (0x3 << 15)); // 2 useful bits from 15 to 16
}
__device__ __host__ inline uint32_t  pack_w18_neutral_bits(uint32_t  w18_nb)
{
	return (w18_nb << 5);
}

__device__ __host__ inline uint32_t  unpack_w19_neutral_bits(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return ((packed_val0 >> 8) & (0x1ff << 6)); // 9 (8 really) useful bits from 6 to 14
}
__device__ __host__ inline uint32_t  pack_w19_neutral_bits(uint32_t  w19_nb)
{
	return (w19_nb << 8);
}

__device__ __host__ inline uint32_t  unpack_w20_neutral_bits(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return (packed_val0 & 0x3fff); // 14 (5 really) useful bits from 0 to 13
}
__device__ __host__ inline uint32_t  pack_w20_neutral_bits(uint32_t  w20_nb)
{
	return w20_nb;
}

__device__ __host__ inline uint32_t  unpack_w21_neutral_bits(uint32_t  packed_val0, uint32_t  packed_val1)
{
	return ((packed_val0 >> 12) & (0x7f << 11)); // 7 (3 really) useful bits from 11 to 17
}
__device__ __host__ inline uint32_t  pack_w21_neutral_bits(uint32_t  w21_nb)
{
	return (w21_nb << 12);
}

/// comprehensive packing for every step ///
__device__ inline uint32_t  pack_update_q18_0(uint32_t  m14, uint32_t  m15)
{
	return (pack_w14_neutral_bits(m14 & W14NBALLM) |
			pack_w15_neutral_bits(m15 & W15NBALLM));

}
__device__ inline uint32_t  pack_update_q18_1(uint32_t  base_idx)
{
	return (pack_base_idx(base_idx));
}

__device__ inline uint32_t  pack_update_q19_0(uint32_t  m14, uint32_t  m15, uint32_t  m16)
{
	return (pack_w14_neutral_bits(m14 & W14NBALLM) |
			pack_w15_neutral_bits(m15 & W15NBALLM) |
			pack_w16_neutral_bits(m16 & W16NBALLM));
}
__device__ inline uint32_t  pack_update_q19_1(uint32_t  m17, uint32_t  base_idx)
{
	return (pack_w17_neutral_bits(m17 & W17NBALLM) |
			pack_base_idx(base_idx));
}

__device__ inline uint32_t  pack_update_q20_0(uint32_t  m14, uint32_t  m15, uint32_t  m16)
{
	return (pack_w14_neutral_bits(m14 & W14NBALLM) |
			pack_w15_neutral_bits(m15 & W15NBALLM) |
			pack_w16_neutral_bits(m16 & W16NBALLM));
}

__device__ inline uint32_t  pack_update_q22_0(uint32_t  m14, uint32_t  m15, uint32_t  m16)
{
	return (pack_w14_neutral_bits(m14 & W14NBALLM) |
			pack_w15_neutral_bits(m15 & W15NBALLM) |
			pack_w16_neutral_bits(m16 & W16NBALLM));
}
__device__ inline uint32_t  pack_update_q22_1(uint32_t  m17, uint32_t  m18, uint32_t  base_idx)
{
	return (pack_w17_neutral_bits(m17 & W17NBALLM) |
			pack_w18_neutral_bits(m18 & W18NBALLM) |
			pack_base_idx(base_idx));
}

__device__ inline uint32_t  pack_update_q23_0(uint32_t  m19)
{
	return pack_w19_neutral_bits(m19 & W19NBALLM);
}
__device__ inline uint32_t  pack_update_q23_1(uint32_t  ext_idx)
{
	return pack_base_idx(ext_idx);
}

__device__ inline uint32_t  pack_update_q24_0(uint32_t  m19, uint32_t  m20, uint32_t  m21)
{
	return (pack_w19_neutral_bits(m19 & W19NBALLM) |
			pack_w20_neutral_bits(m20 & W20NBALLM) |
			pack_w21_neutral_bits(m21 & W21NBALLM));
}
__device__ inline uint32_t  pack_update_q24_1(uint32_t  ext_idx)
{
	return pack_base_idx(ext_idx);
}

__device__ inline uint32_t  pack_update_q27_0(uint32_t  m19, uint32_t  m20, uint32_t  m21)
{
	return (pack_w19_neutral_bits(m19 & W19NBALLM) |
			pack_w20_neutral_bits(m20 & W20NBALLM) |
			pack_w21_neutral_bits(m21 & W21NBALLM));
}
__device__ inline uint32_t  pack_update_q27_1(uint32_t  ext_idx)
{
	return pack_base_idx(ext_idx);
}


/* *** SHA1 FUNCTIONS **********************************
 */
__host__ __device__ inline uint32_t  sha1_round1(uint32_t  a, uint32_t  b, uint32_t  c, uint32_t  d, uint32_t  e, uint32_t  m)
{
	a = rotate_left (a, 5);
	c = rotate_right(c, 2);
	d = rotate_right(d, 2);
	e = rotate_right(e, 2);

	return a + sha1_f1(b, c, d) + e + m + 0x5A827999;
}

__host__ __device__ inline uint32_t  sha1_round2(uint32_t  a, uint32_t  b, uint32_t  c, uint32_t  d, uint32_t  e, uint32_t  m)
{
	a = rotate_left (a, 5);
	c = rotate_right(c, 2);
	d = rotate_right(d, 2);
	e = rotate_right(e, 2);

	return a + sha1_f2(b, c, d) + e + m + 0x6ED9EBA1;
}

__host__ __device__ inline uint32_t  sha1_round3(uint32_t  a, uint32_t  b, uint32_t  c, uint32_t  d, uint32_t  e, uint32_t  m)
{
	a = rotate_left (a, 5);
	c = rotate_right(c, 2);
	d = rotate_right(d, 2);
	e = rotate_right(e, 2);

	return a + sha1_f3(b, c, d) + e + m + 0x8F1BBCDC;
}

__host__ __device__ inline uint32_t  sha1_mess(uint32_t  m_3, uint32_t  m_8, uint32_t  m_14, uint32_t  m_16)
{
	return rotate_left(m_3 ^ m_8 ^ m_14 ^ m_16, 1);
}

#define NEXT_NB(a,mask) { (a) -= 1; (a) &= mask;}


__device__ void stepQ18(uint32_t  base_idx)
{
	PERF_START_COUNTER(18);

	using namespace dev;

	/// fetch the base solution
	uint32_t  q12 = BASESOLBUF.get<0>(base_idx);
	uint32_t  q13 = BASESOLBUF.get<1>(base_idx);
	uint32_t  q14 = BASESOLBUF.get<2>(base_idx);
	uint32_t  q15 = BASESOLBUF.get<3>(base_idx);
	uint32_t  q16 = BASESOLBUF.get<4>(base_idx);
	uint32_t  q17 = BASESOLBUF.get<5>(base_idx);
	uint32_t  m14 = BASESOLBUF.get<14>(base_idx);
	uint32_t  m15 = BASESOLBUF.get<15>(base_idx);
	uint32_t  m16 = BASESOLBUF.get<16>(base_idx);
	uint32_t  m17 = BASESOLBUF.get<17>(base_idx);

	/// loop on the neutral bits
    /// W14: ..................xxxxx......... (5 of 'em)
	/// W15: ...............x.x.............. (2 of 'em)

	uint32_t  w14_q18_nb = 0;
//#pragma unroll
	for (unsigned j = 0; j < 32; j++)
	{
		NEXT_NB(w14_q18_nb, W14NBQ18M);

		/// start to compute the modified state
		m14 &= ~W14NBQ18M; // set some neutral bits
		m14 |= w14_q18_nb;

		q15 += w14_q18_nb; // fast compute of the modified q15
		uint32_t  newq16 = q16 + rotate_left(w14_q18_nb, 5); // partial " q16

		// some conditions on q16, q17 may now have been violated...
		// check delayed until their full value's known

		uint32_t  w15_q18_nb = 0;
//#pragma unroll
		for (unsigned i = 0; i < 4; i++)
		{
			NEXT_NB(w15_q18_nb, W15NBQ18M);

			m15 &= ~W15NBQ18M;
			m15 |= w15_q18_nb;

			newq16 += w15_q18_nb; // complete the computation of q16

			// check on q16
			bool valid_sol = (0 == ((newq16 ^ q16) & Qcondmask[QOFF + 16]));

			// compute and check up to q18!!
			uint32_t  newq17 = sha1_round1(newq16, q15, q14, q13, q12, m16);
			uint32_t  newq18 = sha1_round1(newq17, newq16, q15, q14, q13, m17);

			uint32_t  q18nessies = Qset1mask[QOFF + 18]	^ (Qprevmask  [QOFF + 18] & newq17)
					   									/* ^ (Qprevrmask [QOFF + 18] & rotate_left(newq17, 30)) */
														/* ^ (Qprev2rmask[QOFF + 18] & rotate_left(newq16, 30)) */
														;
			valid_sol &= (0 == ((newq17 ^ q17) & Qcondmask[QOFF + 17]));
			valid_sol &= (0 == ((newq18 ^ q18nessies) & Qcondmask[QOFF + 18]));

			// the solution is made of the base index and the neutral bits of Q18's solution on w14, w15
			uint32_t  sol_val_0 = pack_update_q18_0(m14, m15);
			uint32_t  sol_val_1 = pack_update_q18_1(base_idx);

//			Q18SOLBUF.write(Q18SOLCTL, valid_sol, sol_val_0, sol_val_1);
			WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q18SOLBUF, Q18SOLCTL);

			newq16 -= w15_q18_nb; // forget about these now
		}
		q15 -= w14_q18_nb; // and these
	}
	WARP_TMP_BUF.flush2(Q18SOLBUF, Q18SOLCTL);
	PERF_STOP_COUNTER(18);
}



__device__ void stepQ19(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(19);
	using namespace dev;

	uint32_t  q18_sol0 = Q18SOLBUF.get<0>(thread_rd_idx);
	uint32_t  q18_sol1 = Q18SOLBUF.get<1>(thread_rd_idx);
	uint32_t  base_idx 	= unpack_base_idx(q18_sol0, q18_sol1);
	uint32_t  w14_sol_nb = unpack_w14_neutral_bits(q18_sol0, q18_sol1);
	uint32_t  w15_sol_nb = unpack_w15_neutral_bits(q18_sol0, q18_sol1);

	/// fetch the base solution and update it to the Q18 solution using the above neutral bits
	uint32_t  q12 = BASESOLBUF.get<0>(base_idx);
	uint32_t  q13 = BASESOLBUF.get<1>(base_idx);
	uint32_t  q14 = BASESOLBUF.get<2>(base_idx);
	uint32_t  q15 = BASESOLBUF.get<3>(base_idx);
	uint32_t  q16 = BASESOLBUF.get<4>(base_idx);
	uint32_t  q17;
	uint32_t  q18;
	uint32_t  m14 = BASESOLBUF.get<14>(base_idx);
	uint32_t  m15 = BASESOLBUF.get<15>(base_idx);
	uint32_t  m16 = BASESOLBUF.get<16>(base_idx);
	uint32_t  m17 = BASESOLBUF.get<17>(base_idx);
	uint32_t  m18 = BASESOLBUF.get<18>(base_idx);

	m14 += w14_sol_nb; // here, + == ^ == |, as all the neutral bit positions are kept to 0 in the base message words
	m15 += w15_sol_nb;
	q15 += w14_sol_nb; // accomodate differences in m14
	q16 += w15_sol_nb; // "                    " in m15
	q16 += rotate_left(w14_sol_nb, 5); // "    " in q15
	q17  = sha1_round1(q16, q15, q14, q13, q12, m16);
	q18  = sha1_round1(q17, q16, q15, q14, q13, m17);

	/// loop on the neutral bits
	/// W14: .......................xxxx..... (4 of 'em)
	/// W15: ..................xxxxxx........ (6 of 'em)
	/// W16: ...............x..x............. (2 of 'em)
	/// W17: ............x................... (1 of 'em)

	uint32_t  w14_q19_nb = 0;
	do
//#pragma unroll
//	for (unsigned l = 0; l < 16; ++l)
	{
		NEXT_NB(w14_q19_nb, W14NBQ19M);

		// start to recompute the previous state
		m14 &= ~W14NBQ19M;
		m14 |= w14_q19_nb;

		q15 += w14_q19_nb;
		uint32_t  newq16 = q16 + rotate_left(w14_q19_nb, 5);

		// some conditions on q16 may now have been violated... check for that
		bool valid_sol = 0 == ((newq16 ^ q16) & Qcondmask[QOFF + 16]);
		// same goes for q17, q18, but the check is delayed

		uint32_t  w15_q19_nb = 0;
//		do
//#pragma unroll
		for (unsigned k = 0; k < 64; ++k)
		{
			NEXT_NB(w15_q19_nb, W15NBQ19M);

			m15 &= ~W15NBQ19M;
			m15 |= w15_q19_nb;

			newq16 += w15_q19_nb;

			// check on q17, q18 delayed
			uint32_t  newq17 = sha1_round1(newq16, q15, q14, q13, q12, m16);

			uint32_t  w16_q19_nb = 0;
//			do
//#pragma unroll
			for (unsigned j = 0; j < 4; ++j)
			{
				NEXT_NB(w16_q19_nb, W16NBQ19M);

				m16 &= ~W16NBQ19M;
				m16 |= w16_q19_nb;

				newq17 += w16_q19_nb;

				// check on q17 because of neutral bits of w14, w15, w16
				valid_sol &= 0 == ((newq17 ^ q17) & Qcondmask[QOFF + 17]);
				// check on q18 delayed

				uint32_t  w17_q19_nb = 0;
//#pragma unroll
				for (unsigned i = 0; i < 2; ++i)
//				do
				{
					NEXT_NB(w17_q19_nb, W17NBQ19M);

					m17 &= ~W17NBQ19M;
					m17 |= w17_q19_nb;

					uint32_t  newq18 = sha1_round1(newq17, newq16, q15, q14, q13, m17);
					uint32_t  newq19 = sha1_round1(newq18, newq17, newq16, q15, q14, m18);

					// check for a brand new solution for q19!!
					// the compiler doesn't remove zero masks... need to do it by hand ><
					uint32_t  q19nessies = Qset1mask[QOFF + 19]	/* ^ (Qprevmask  [QOFF + 19] & newq18) */
					   											^ (Qprevrmask [QOFF + 19] & rotate_left(newq18, 30))
																/* ^ (Qprev2rmask[QOFF + 19] & rotate_left(newq17, 30)) */
																;
					// check on q18 because of neutral bits of w14, w15, w16, w17
					valid_sol &= 0 == ((newq18 ^ q18) & Qcondmask[QOFF + 18]);
					valid_sol &= 0 == ((newq19 ^ q19nessies) & Qcondmask[QOFF + 19]);

					// the solution is made of the base index and the neutral bits of Q18's solution on w14, w15
					// plus the neutral bits on w14, w15, w16, w17
					uint32_t  sol_val_0 = pack_update_q19_0(m14, m15, m16);
					uint32_t  sol_val_1 = pack_update_q19_1(m17, base_idx);
					WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q19SOLBUF, Q19SOLCTL);
				}
//				} while (w17_q19_nb != 0);

				newq17 -= w16_q19_nb;
			}
//			} while (w16_q19_nb != 0);

			newq16 -= w15_q19_nb;
//		} while (w15_q19_nb != 0);
		}

		q15 -= w14_q19_nb;
	} while (w14_q19_nb != 0);
//	}
	WARP_TMP_BUF.flush2(Q19SOLBUF, Q19SOLCTL);
	PERF_STOP_COUNTER(19);
}



__device__ void stepQ20(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(20);
	using namespace dev;

	uint32_t  q19_sol0 = Q19SOLBUF.get<0>(thread_rd_idx);
	uint32_t  q19_sol1 = Q19SOLBUF.get<1>(thread_rd_idx);
	uint32_t  base_idx 	= unpack_base_idx(q19_sol0, q19_sol1);
	uint32_t  w14_sol_nb = unpack_w14_neutral_bits(q19_sol0, q19_sol1);
	uint32_t  w15_sol_nb = unpack_w15_neutral_bits(q19_sol0, q19_sol1);
	uint32_t  w16_sol_nb = unpack_w16_neutral_bits(q19_sol0, q19_sol1);
	uint32_t  w17_sol_nb = unpack_w17_neutral_bits(q19_sol0, q19_sol1);

	/// fetch the base solution and update it to the Q19 solution using the above neutral bits
	uint32_t  q12 = BASESOLBUF.get<0>(base_idx);
	uint32_t  q13 = BASESOLBUF.get<1>(base_idx);
	uint32_t  q14 = BASESOLBUF.get<2>(base_idx);
	uint32_t  q15 = BASESOLBUF.get<3>(base_idx);
	uint32_t  q16 = BASESOLBUF.get<4>(base_idx);
	uint32_t  q17;
	uint32_t  q18;
	uint32_t  q19;
	uint32_t  m14 = BASESOLBUF.get<14>(base_idx);
	uint32_t  m15 = BASESOLBUF.get<15>(base_idx);
	uint32_t  m16 = BASESOLBUF.get<16>(base_idx);
	uint32_t  m17 = BASESOLBUF.get<17>(base_idx);
	uint32_t  m18 = BASESOLBUF.get<18>(base_idx);
	uint32_t  m19 = BASESOLBUF.get<19>(base_idx);

	m14 += w14_sol_nb;
	m15 += w15_sol_nb;
	m16 += w16_sol_nb;
	m17 += w17_sol_nb;

	q15 += w14_sol_nb;
	q16 += w15_sol_nb;
	q16 += rotate_left(w14_sol_nb, 5);
	q17  = sha1_round1(q16, q15, q14, q13, q12, m16);
	q18  = sha1_round1(q17, q16, q15, q14, q13, m17);
	q19  = sha1_round1(q18, q17, q16, q15, q14, m18);

	/// loop on the neutral bits
	/// W15: ........................xxx..... (3 of 'em)
	/// W16: ....................xxx......... (3 of 'em)

	uint32_t  w15_q20_nb = 0;
//#pragma unroll
	for (unsigned j = 0; j < 8; j++)
	{
		NEXT_NB(w15_q20_nb, W15NBQ20M);

		m15 &= ~W15NBQ20M;
		m15 |= w15_q20_nb;

		q16 += w15_q20_nb; // fast compute of the modified q16
		q17 += rotate_left(w15_q20_nb, 5); // partial "	   q17

		// no conditions can go wrong at this point

		uint32_t  w16_q20_nb = 0;
//#pragma unroll
		for (unsigned i = 0; i < 8; i++)
		{
			NEXT_NB(w16_q20_nb, W16NBQ20M);

			m16 &= ~W16NBQ20M;
			m16 |= w16_q20_nb;

			q17 += w16_q20_nb; // complete the computation of q17

			// compute and check up to q20!!
			uint32_t  newq18 = sha1_round1(q17, q16, q15, q14, q13, m17);
			uint32_t  newq19 = sha1_round1(newq18, q17, q16, q15, q14, m18);
			uint32_t  newq20 = sha1_round1(newq19, newq18, q17, q16, q15, m19);

			uint32_t  q20nessies = Qset1mask[QOFF + 20]	/* ^ (Qprevmask  [QOFF + 20] & newq19) */
					   									^ (Qprevrmask [QOFF + 20] & rotate_left(newq19, 30))
														^ (Qprev2rmask[QOFF + 20] & rotate_left(newq18, 30))
														;
			bool valid_sol = (0 == ((newq18 ^ q18) & Qcondmask[QOFF + 18]));
			valid_sol &= (0 == ((newq19 ^ q19) & Qcondmask[QOFF + 19]));
			valid_sol &= (0 == ((newq20 ^ q20nessies) & Qcondmask[QOFF + 20]));

			// the solution is made of the base index and the neutral bits of Q19's solution on w14, w15, w16, w17
			// plus the neutral bits on w15, w16
			uint32_t  sol_val_0 = pack_update_q20_0(m14, m15, m16);
			// no need to change q19_sol1
			WARP_TMP_BUF.write2(valid_sol, sol_val_0, q19_sol1, Q20SOLBUF, Q20SOLCTL);

			q17 -= w16_q20_nb;
		}
		q16 -= w15_q20_nb;
		q17 -= rotate_left(w15_q20_nb, 5);
	}
	WARP_TMP_BUF.flush2(Q20SOLBUF, Q20SOLCTL);
	PERF_STOP_COUNTER(20);
}

/* Flips w20[14] & w20[16] condtionally on the use of certain neutral bits
 * in w15 & w16
*/
/// w20[14]: .........x.x.....T...x.......... (w15[16], w15[14], w16[16])
#define W2014FILTER (1 << 14)
/// w20[16]: .........x.....T................ (w15[16])
#define W2016FILTER (1 << 16)
__device__ inline uint32_t  w20_nb_partial_correction(uint32_t  q20_sol0)
{
	uint32_t  flip2014 = ((q20_sol0 >> 8) ^ (q20_sol0 >> 6) ^ (q20_sol0 << 4)) & W2014FILTER;
	uint32_t  flip2016 = (q20_sol0 >> 6) & W2016FILTER;

	return (flip2014 ^ flip2016);
}
#define W2014FLIPM2 0x00008000 // ................x............... (w18[15], unpacked)

__device__ void stepQ21(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(21);
	using namespace dev;

	uint32_t  q20_sol0 = Q20SOLBUF.get<0>(thread_rd_idx);
	uint32_t  q20_sol1 = Q20SOLBUF.get<1>(thread_rd_idx);
	uint32_t  base_idx   = unpack_base_idx        (q20_sol0, q20_sol1);
	uint32_t  w14_sol_nb = unpack_w14_neutral_bits(q20_sol0, q20_sol1);
	uint32_t  w15_sol_nb = unpack_w15_neutral_bits(q20_sol0, q20_sol1);
	uint32_t  w16_sol_nb = unpack_w16_neutral_bits(q20_sol0, q20_sol1);
	uint32_t  w17_sol_nb = unpack_w17_neutral_bits(q20_sol0, q20_sol1);

	/// fetch the base solution and update it to the Q20 solution using the above neutral bits
	uint32_t  q12 = BASESOLBUF.get<0>(base_idx);
	uint32_t  q13 = BASESOLBUF.get<1>(base_idx);
	uint32_t  q14 = BASESOLBUF.get<2>(base_idx);
	uint32_t  q15 = BASESOLBUF.get<3>(base_idx);
	uint32_t  q16 = BASESOLBUF.get<4>(base_idx);
	uint32_t  q17;
	uint32_t  q18;
	uint32_t  q19;
	uint32_t  q20;
	uint32_t  m14 = BASESOLBUF.get<14>(base_idx);
	uint32_t  m15 = BASESOLBUF.get<15>(base_idx);
	uint32_t  m16 = BASESOLBUF.get<16>(base_idx);
	uint32_t  m17 = BASESOLBUF.get<17>(base_idx);
	uint32_t  m18 = BASESOLBUF.get<18>(base_idx);
	uint32_t  m19 = BASESOLBUF.get<19>(base_idx);
	uint32_t  m20 = BASESOLBUF.get<20>(base_idx);
	uint32_t  m21 = BASESOLBUF.get<21>(base_idx);

	m14 += w14_sol_nb;
	m15 += w15_sol_nb;
	m16 += w16_sol_nb;
	m17 += w17_sol_nb;
	m20 ^= w20_nb_partial_correction(q20_sol0); // correct the value according to previous nbs, if necessary

	q15 += w14_sol_nb;
	q16 += w15_sol_nb;
	q16 += rotate_left(w14_sol_nb, 5);
	q17  = sha1_round1(q16, q15, q14, q13, q12, m16);
	q18  = sha1_round1(q17, q16, q15, q14, q13, m17);
	q19  = sha1_round1(q18, q17, q16, q15, q14, m18);
	q20  = sha1_round1(q19, q18, q17, q16, q15, m19);

	/// loop on the neutral bits
	/// W16 .......................xxx...... (3 of 'em)
	/// W17 ..................xxxx.......... (4 of 'em)
	/// W18 ...............xx............... (2 of 'em)

	uint32_t  w16_q21_nb = 0;
//#pragma unroll
	for (unsigned k = 0; k < 8; k++)
	{
		NEXT_NB(w16_q21_nb, W16NBQ21M);

		m16 &= ~W16NBQ21M;
		m16 |= w16_q21_nb;

		q17 += w16_q21_nb; // fast compute of the modified q17
		q18 += rotate_left(w16_q21_nb, 5); // partial "	   q18

		// no conditions can go wrong at this point

		uint32_t  w17_q21_nb = 0;
//#pragma unroll
		for (unsigned j = 0; j < 16; j++)
		{
			NEXT_NB(w17_q21_nb, W17NBQ21M);

			m17 &= ~W17NBQ21M;
			m17 |= w17_q21_nb;

			q18 += w17_q21_nb; // complete the computation of q18

			// no conditions can go wrong at this point

			uint32_t  w18_q21_nb = 0;
//#pragma unroll
			for (unsigned i = 0; i < 4; i++)
			{
				NEXT_NB(w18_q21_nb, W18NBQ21M);

				m18 &= ~W18NBQ21M;
				m18 |= w18_q21_nb;

				m20 ^= (m18 & W2014FLIPM2) >> 1; // second correction for m20

				// compute and check up to q22!!
				uint32_t  newq19 = sha1_round1(q18, q17, q16, q15, q14, m18);
				uint32_t  newq20 = sha1_round1(newq19, q18, q17, q16, q15, m19);
				uint32_t  newq21 = sha1_round2(newq20, newq19, q18, q17, q16, m20);
				uint32_t  newq22 = sha1_round2(newq21, newq20, newq19, q18, q17, m21);

				uint32_t  q21nessies = Qset1mask[QOFF + 21]	/* ^ (Qprevmask  [QOFF + 21] & newq20) */
															^ (Qprevrmask [QOFF + 21] & rotate_left(newq20, 30))
															/* ^ (Qprev2rmask[QOFF + 21] & rotate_left(newq19, 30)) */
															;
				uint32_t  q22nessies = Qset1mask[QOFF + 22]	/* ^ (Qprevmask  [QOFF + 22] & newq21) */
															/* ^ (Qprevrmask [QOFF + 22] & rotate_left(newq21, 30)) */
															^ (Qprev2rmask[QOFF + 22] & rotate_left(newq20, 30))
															;
				bool valid_sol = (0 == ((newq19 ^ q19) & Qcondmask[QOFF + 19]));
				valid_sol &= (0 == ((newq20 ^ q20) & Qcondmask[QOFF + 20]));
				valid_sol &= (0 == ((newq21 ^ q21nessies) & Qcondmask[QOFF + 21]));
				valid_sol &= (0 == ((newq22 ^ q22nessies) & Qcondmask[QOFF + 22]));

#ifndef PAC22
				// write an extended base solution made of q17..q21,m14..m18,m20,base_idx
				Q22SOLBUF.write(Q22SOLCTL, valid_sol, q17, q18, newq19, newq20, newq21, m14, m15, m16, m17, m18, m20, base_idx);
#else
				// For now, try something different with a packed sol, later unpacked to the extended base
				// in an auxiliary function
				uint32_t  sol_val_0 = pack_update_q22_0(m14, m15, m16);
				uint32_t  sol_val_1 = pack_update_q22_1(m17, m18, base_idx);
				WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q22PACBUF, Q22PACCTL);
#endif

				m20 ^= (m18 & W2014FLIPM2) >> 1; // remove second correction for m20
			}
			q18 -= w17_q21_nb;
		}
		q17 -= w16_q21_nb;
		q18 -= rotate_left(w16_q21_nb, 5);
	}
#ifdef PAC22
	WARP_TMP_BUF.flush2(Q22PACBUF, Q22PACCTL);
#endif
	PERF_STOP_COUNTER(21);
}

#ifdef PAC22
// auxiliary function for stepQ23: batch reconstruction of extended base solutions from
// compressed Q22 ones
__device__ void stepQ23_aux(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(22);
	using namespace dev;

	uint32_t  q22_sol0 	= Q22PACBUF.get<0>(thread_rd_idx);
	uint32_t  q22_sol1 	= Q22PACBUF.get<1>(thread_rd_idx);
	uint32_t  base_idx  	= unpack_base_idx(q22_sol0, q22_sol1);
	uint32_t  w14_sol_nb = unpack_w14_neutral_bits(q22_sol0, q22_sol1);
	uint32_t  w15_sol_nb = unpack_w15_neutral_bits(q22_sol0, q22_sol1);
	uint32_t  w16_sol_nb = unpack_w16_neutral_bits(q22_sol0, q22_sol1);
	uint32_t  w17_sol_nb = unpack_w17_neutral_bits(q22_sol0, q22_sol1);
	uint32_t  w18_sol_nb = unpack_w18_neutral_bits(q22_sol0, q22_sol1);

	/// fetch the base solution and update it to the Q22 solution using the above neutral bits
	uint32_t  q12 = BASESOLBUF.get<0>(base_idx);
	uint32_t  q13 = BASESOLBUF.get<1>(base_idx);
	uint32_t  q14 = BASESOLBUF.get<2>(base_idx);
	uint32_t  q15 = BASESOLBUF.get<3>(base_idx);
	uint32_t  q16 = BASESOLBUF.get<4>(base_idx);
	uint32_t  m14 = BASESOLBUF.get<14>(base_idx);
	uint32_t  m15 = BASESOLBUF.get<15>(base_idx);
	uint32_t  m16 = BASESOLBUF.get<16>(base_idx);
	uint32_t  m17 = BASESOLBUF.get<17>(base_idx);
	uint32_t  m18 = BASESOLBUF.get<18>(base_idx);
	uint32_t  m19 = BASESOLBUF.get<19>(base_idx);
	uint32_t  m20 = BASESOLBUF.get<20>(base_idx);

	m14 += w14_sol_nb;
	m15 += w15_sol_nb;
	m16 += w16_sol_nb;
	m17 += w17_sol_nb;
	m18 += w18_sol_nb;
	m20 ^= w20_nb_partial_correction(q22_sol0); // correct the value according to previous nbs, if necessary
	m20 ^= (m18 & W2014FLIPM2) >> 1; // second correction for m20

	q15 += w14_sol_nb;
	q16 += w15_sol_nb;
	q16 += rotate_left(w14_sol_nb, 5);
	uint32_t  q17  = sha1_round1(q16, q15, q14, q13, q12, m16);
	uint32_t  q18  = sha1_round1(q17, q16, q15, q14, q13, m17);
	uint32_t  q19  = sha1_round1(q18, q17, q16, q15, q14, m18);
	uint32_t  q20  = sha1_round1(q19, q18, q17, q16, q15, m19);
	uint32_t  q21  = sha1_round2(q20, q19, q18, q17, q16, m20);

	// why not copying q22 again?? Not that it should matter too much here
	Q22SOLBUF.write(Q22SOLCTL, true, q17, q18, q19, q20, q21, m14, m15, m16, m17, m18, m20, base_idx);
	PERF_STOP_COUNTER(22);
}
#endif

__device__ void stepQ23(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(23);
	using namespace dev;

	/// fetch the base solution, extended
	uint32_t  base_idx = Q22SOLBUF.get<11>(thread_rd_idx);
	uint32_t  q17 = Q22SOLBUF.get<0>(thread_rd_idx);
	uint32_t  q18 = Q22SOLBUF.get<1>(thread_rd_idx);
	uint32_t  q19 = Q22SOLBUF.get<2>(thread_rd_idx);
	uint32_t  q20 = Q22SOLBUF.get<3>(thread_rd_idx);
	uint32_t  q21 = Q22SOLBUF.get<4>(thread_rd_idx);
//	uint32_t  q22;
	uint32_t  m6  = BASESOLBUF.get<6>(base_idx);
	uint32_t  m8  = BASESOLBUF.get<8>(base_idx);
	uint32_t  m19 = BASESOLBUF.get<19>(base_idx);
	uint32_t  m21 = BASESOLBUF.get<21>(base_idx);
	uint32_t  m14 = Q22SOLBUF.get<5>(thread_rd_idx);
	uint32_t  m22;

	uint32_t  q22  = sha1_round2(q21, q20, q19, q18, q17, m21);

	/// loop on the neutral bits
	/// W19 .................x.xxxx......... (5 of 'em)

	uint32_t  w19_q23_nb = 0;
//#pragma unroll
	for (unsigned i = 0; i < 32; i++)
	{
		NEXT_NB(w19_q23_nb, W19NBQ23M);

		m19 &= ~W19NBQ23M;
		m19 |= w19_q23_nb;
		m22 = sha1_mess(m19, m14, m8, m6); // could be partially cached before, maybe t'would be better?

		// compute and check up to q23!!
		uint32_t  newq20 = q20 + w19_q23_nb;
		uint32_t  newq21 = q21 + rotate_left(w19_q23_nb, 5);
		uint32_t  newq22 = sha1_round2(newq21, newq20, q19, q18, q17, m21);
		uint32_t  newq23 = sha1_round2(newq22, newq21, newq20, q19, q18, m22);

		uint32_t  q23nessies = Qset1mask[QOFF + 23]	/* ^ (Qprevmask  [QOFF + 23] & newq22) */
													^ (Qprevrmask [QOFF + 23] & rotate_left(newq22, 30))
													/* ^ (Qprev2rmask[QOFF + 23] & rotate_left(newq21, 30)) */
													;
		bool valid_sol = (0 == ((newq21 ^ q21) & Qcondmask[QOFF + 21]));
		valid_sol &= (0 == ((newq22 ^ q22) & Qcondmask[QOFF + 22]));
		valid_sol &= (0 == ((newq23 ^ q23nessies) & Qcondmask[QOFF + 23]));

		// the solution is made of the extended base solution index and the neutral bits on w19
		uint32_t  sol_val_0 = pack_update_q23_0(m19);
		uint32_t  sol_val_1 = pack_update_q23_1(thread_rd_idx);

		WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q23SOLBUF, Q23SOLCTL);

	}
	WARP_TMP_BUF.flush2(Q23SOLBUF, Q23SOLCTL);
	PERF_STOP_COUNTER(23);
}

__device__ void stepQ24(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(24);
	using namespace dev;

	/// fetch the base solution, extended
	uint32_t  q23_sol0 = Q23SOLBUF.get<0>(thread_rd_idx);
	uint32_t  q23_sol1 = Q23SOLBUF.get<1>(thread_rd_idx);

	uint32_t  ext_idx 	= unpack_ext_idx(q23_sol0, q23_sol1);
	uint32_t  w19_sol_nb = unpack_w19_neutral_bits(q23_sol0, q23_sol1);

	uint32_t  base_idx = Q22SOLBUF.get<11>(ext_idx);
	uint32_t  q17 = Q22SOLBUF.get<0>(ext_idx);
	uint32_t  q18 = Q22SOLBUF.get<1>(ext_idx);
	uint32_t  q19 = Q22SOLBUF.get<2>(ext_idx);
	uint32_t  q20 = Q22SOLBUF.get<3>(ext_idx);
	uint32_t  q21 = Q22SOLBUF.get<4>(ext_idx);
	uint32_t  q22;
	uint32_t  q23;
	uint32_t  m6  = BASESOLBUF.get<6>(base_idx);
	uint32_t  m7  = BASESOLBUF.get<7>(base_idx);
	uint32_t  m8  = BASESOLBUF.get<8>(base_idx);
	uint32_t  m9  = BASESOLBUF.get<9>(base_idx);
	uint32_t  m19 = BASESOLBUF.get<19>(base_idx);
	uint32_t  m21 = BASESOLBUF.get<21>(base_idx);
	uint32_t  m14 = Q22SOLBUF.get<5>(ext_idx);
	uint32_t  m15 = Q22SOLBUF.get<6>(ext_idx);
	uint32_t  m20 = Q22SOLBUF.get<10>(ext_idx);
	uint32_t  m22;
	uint32_t  m23;

	m19 += w19_sol_nb;
	m22  = sha1_mess(m19, m14, m8, m6);

	q20 += w19_sol_nb;
	q21 += rotate_left(w19_sol_nb, 5);
	q22  = sha1_round2(q21, q20, q19, q18, q17, m21);
	q23  = sha1_round2(q22, q21, q20, q19, q18, m22);

	/// loop on the neutral bits
	/// W19 ........................xx...... (2 of 'em)
	/// W20 ...................xx........... (2 of 'em)
	/// W21 ..............xx................ (2 of 'em)

	uint32_t  w19_q24_nb = 0;
//#pragma unroll
	for (unsigned k = 0; k < 4; k++)
	{
		NEXT_NB(w19_q24_nb, W19NBQ24M);

		m19 &= ~W19NBQ24M;
		m19 |= w19_q24_nb;

		m22 ^= rotate_left(w19_q24_nb, 1); // a shift left would do to
		q20 += w19_q24_nb;
		q21 += rotate_left(w19_q24_nb, 5);

		// no checks to do

		uint32_t  w20_q24_nb = 0;
//#pragma unroll
		for (unsigned j = 0; j < 4; j++)
		{
			NEXT_NB(w20_q24_nb, W20NBQ24M);

			m20 &= ~W20NBQ24M;
			m20 |= w20_q24_nb;
			m23 = sha1_mess(m20, m15, m9, m7); // could be partially cached

			q21 += w20_q24_nb;
			uint32_t  newq22 = sha1_round2(q21, q20, q19, q18, q17, m21);

			// checks on q22 delayed

			uint32_t  w21_q24_nb = 0;
			for (unsigned i = 0; i < 4; i++)
			{
				NEXT_NB(w21_q24_nb, W21NBQ24M);

				m21 &= ~W21NBQ24M;
				m21 |= w21_q24_nb;

				newq22 += w21_q24_nb;

				// compute and check up to q24!!
				uint32_t  newq23 = sha1_round2(newq22, q21, q20, q19, q18, m22);
				uint32_t  newq24 = sha1_round2(newq23, newq22, q21, q20, q19, m23);

				uint32_t  q24nessies = Qset1mask[QOFF + 24]	/* ^ (Qprevmask  [QOFF + 24] & newq23) */
															/* ^ (Qprevrmask [QOFF + 24] & rotate_left(newq23, 30)) */
															^ (Qprev2rmask[QOFF + 24] & rotate_left(newq22, 30))
															;

				bool valid_sol = (0 == ((newq22 ^ q22) & Qcondmask[QOFF + 22]));
				valid_sol &= (0 == ((newq23 ^ q23) & Qcondmask[QOFF + 23]));
				valid_sol &= (0 == ((newq24 ^ q24nessies) & Qcondmask[QOFF + 24]));

				// the solution is made of the extended base solution index and the neutral bits on w19, w20, w21
				uint32_t  sol_val_0 = pack_update_q24_0(m19, m20, m21);
				uint32_t  sol_val_1 = pack_update_q24_1(ext_idx);

				WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q24SOLBUF, Q24SOLCTL);

				newq22 -= w21_q24_nb;
			}
			q21 -= w20_q24_nb;
		}
		m22 ^= rotate_left(w19_q24_nb, 1);
		q20 -= w19_q24_nb;
		q21 -= rotate_left(w19_q24_nb, 5);
	}
	WARP_TMP_BUF.flush2(Q24SOLBUF, Q24SOLCTL);
	PERF_STOP_COUNTER(24);
}

__device__ void stepQ256(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(25);
	using namespace dev;

	/// fetch the base solution, extended
	uint32_t  q24_sol0 = Q24SOLBUF.get<0>(thread_rd_idx);
	uint32_t  q24_sol1 = Q24SOLBUF.get<1>(thread_rd_idx);

	uint32_t  ext_idx 	= unpack_ext_idx(q24_sol0, q24_sol1);
	uint32_t  w19_sol_nb = unpack_w19_neutral_bits(q24_sol0, q24_sol1);
	uint32_t  w20_sol_nb = unpack_w20_neutral_bits(q24_sol0, q24_sol1);
	uint32_t  w21_sol_nb = unpack_w21_neutral_bits(q24_sol0, q24_sol1);

	uint32_t  base_idx = Q22SOLBUF.get<11>(ext_idx);
	uint32_t  q17 = Q22SOLBUF.get<0>(ext_idx);
	uint32_t  q18 = Q22SOLBUF.get<1>(ext_idx);
	uint32_t  q19 = Q22SOLBUF.get<2>(ext_idx);
	uint32_t  q20 = Q22SOLBUF.get<3>(ext_idx);
	uint32_t  q21 = Q22SOLBUF.get<4>(ext_idx);
	uint32_t  m6  = BASESOLBUF.get<6>(base_idx);
	uint32_t  m7  = BASESOLBUF.get<7>(base_idx);
	uint32_t  m8  = BASESOLBUF.get<8>(base_idx);
	uint32_t  m9  = BASESOLBUF.get<9>(base_idx);
	uint32_t  m10 = BASESOLBUF.get<10>(base_idx);
	uint32_t  m11 = BASESOLBUF.get<11>(base_idx);
	uint32_t  m12 = BASESOLBUF.get<12>(base_idx);
	uint32_t  m19 = BASESOLBUF.get<19>(base_idx);
	uint32_t  m21 = BASESOLBUF.get<21>(base_idx);
	uint32_t  m14 = Q22SOLBUF.get<5>(ext_idx);
	uint32_t  m15 = Q22SOLBUF.get<6>(ext_idx);
	uint32_t  m16 = Q22SOLBUF.get<7>(ext_idx);
	uint32_t  m17 = Q22SOLBUF.get<8>(ext_idx);
	uint32_t  m18 = Q22SOLBUF.get<9>(ext_idx);
	uint32_t  m20 = Q22SOLBUF.get<10>(ext_idx);

	m19 += w19_sol_nb;
	m20 += w20_sol_nb;
	m21 += w21_sol_nb;
	uint32_t  m22  = sha1_mess(m19, m14, m8, m6);
	uint32_t  m23  = sha1_mess(m20, m15, m9, m7);

	q20 += w19_sol_nb;
	q21 += rotate_left(w19_sol_nb, 5) + w20_sol_nb;
	uint32_t  q22  = sha1_round2(q21, q20, q19, q18, q17, m21);
	uint32_t  q23  = sha1_round2(q22, q21, q20, q19, q18, m22);
	uint32_t  q24  = sha1_round2(q23, q22, q21, q20, q19, m23);

	/// loop on the neutral bits (Q25 only)
	/// W19 .......................x........ (1 of 'em)
	/// W20 ..................x......x...... (2 of 'em)
	/// W21 ....................x........... (1 of 'em)

	uint32_t  w19_q25_nb = 0;
//#pragma unroll
	for (unsigned k = 0; k < 2; k++)
	{
		NEXT_NB(w19_q25_nb, W19NBQ25M);

		m19 &= ~W19NBQ25M;
		m19 |= w19_q25_nb;

		m22 ^= rotate_left(w19_q25_nb, 1); // a shift left would do to
		q20 += w19_q25_nb;
		q21 += rotate_left(w19_q25_nb, 5);

		// no checks to do

		uint32_t  w20_q25_nb = 0;
//#pragma unroll
		for (unsigned j = 0; j < 8; j++)
		{
			NEXT_NB(w20_q25_nb, W20NBQ25M|W20NBQ26M);

			m20 &= ~(W20NBQ25M|W20NBQ26M);
			m20 |= w20_q25_nb;

			m23 ^= rotate_left(w20_q25_nb, 1);
			q21 += w20_q25_nb;
			uint32_t  newq22 = sha1_round2(q21, q20, q19, q18, q17, m21);

			// checks on q22 delayed

			uint32_t  w21_q25_nb = 0;
//#pragma unroll
			for (unsigned i = 0; i < 2; i++)
			{
				NEXT_NB(w21_q25_nb, W21NBQ25M);

				m21 &= ~W21NBQ25M;
				m21 |= w21_q25_nb;

				newq22 += w21_q25_nb;

#define QSET1M25 0x28000000
#define QSET1M26 0x88000000
#define QSET1M27 0x60000000

#define QPREV26 0x00000001

#define QPREVR25 0x08000000
#define QPREVR27 0x08000000

#define QPREV2R26 0x08000000

#define QCOND25 0x28000000
#define QCOND26 0x88000001
#define QCOND27 0x68000000

				// compute and check up to q25!!
				uint32_t  m24 = sha1_mess(m21, m16, m10, m8);
				uint32_t  newq23 = sha1_round2(newq22, q21, q20, q19, q18, m22);
				uint32_t  newq24 = sha1_round2(newq23, newq22, q21, q20, q19, m23);
				uint32_t  newq25 = sha1_round2(newq24, newq23, newq22, q21, q20, m24);

//				uint32_t  q25nessies = Qset1mask[QOFF + 25]	/* ^ (Qprevmask  [QOFF + 25] & newq24) */
//															^ (Qprevrmask [QOFF + 25] & rotate_left(newq24, 30))
//															/* ^ (Qprev2rmask[QOFF + 25] & rotate_left(newq23, 30)) */
//															;
				uint32_t  q25nessies = QSET1M25 ^ (QPREVR25 & rotate_left(newq24, 30));

				bool valid_sol = (0 == ((newq22 ^ q22) & Qcondmask[QOFF + 22]));
				valid_sol &= (0 == ((newq23 ^ q23) & Qcondmask[QOFF + 23]));
				valid_sol &= (0 == ((newq24 ^ q24) & Qcondmask[QOFF + 24]));
				valid_sol &= (0 == ((newq25 ^ q25nessies) & QCOND25));

				// if still valid, try 26/27 with its inlined neutral bit

				uint32_t  m25 = sha1_mess(m22, m17, m11, m9);
				uint32_t  newq26 = sha1_round2(newq25, newq24, newq23, newq22, q21, m25);
//				uint32_t  q26nessies = Qset1mask[QOFF + 26]	^ (Qprevmask  [QOFF + 26] & newq25)
//									/* ^ (Qprevrmask [QOFF + 26] & rotate_left(newq25, 30)) */
//									^ (Qprev2rmask[QOFF + 26] & rotate_left(newq24, 30))
//																;
				uint32_t  q26nessies = QSET1M26 ^ (QPREV26 & newq25) ^ (QPREV2R26 & rotate_left(newq24, 30));

				valid_sol &= (0 == ((newq26 ^ q26nessies) & QCOND26));
				uint32_t  sol_val_0;
				uint32_t  sol_val_1;
//				if (valid_sol)
//				{
					uint32_t  m26 = sha1_mess(m23, m18, m12, m10);
					uint32_t  newq27 = sha1_round2(newq26, newq25, newq24, newq23, newq22, m26);

//					uint32_t  q27nessies = Qset1mask[QOFF + 27]	/* ^ (Qprevmask  [QOFF + 27] & newq26) */
//																^ (Qprevrmask [QOFF + 27] & rotate_left(newq26, 30))
//																/* ^ (Qprev2rmask[QOFF + 27] & rotate_left(newq26, 30)) */
//																;
					uint32_t  q27nessies = QSET1M27 ^ (QPREVR27 & rotate_left(newq26, 30));

					valid_sol &= (0 == ((newq27 ^ q27nessies) & QCOND27));
					// the solution is made of the extended base and the neutral bits on w19,w20,w21
					sol_val_0 = pack_update_q27_0(m19, m20, m21);
					sol_val_1 = pack_update_q27_1(ext_idx);
//				}
				// All threads need to participate in the write
				WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q27SOLBUF, Q27SOLCTL);

				newq22 -= w21_q25_nb;
			}
			m23 ^= rotate_left(w20_q25_nb, 1);
			q21 -= w20_q25_nb;
		}
		m22 ^= rotate_left(w19_q25_nb, 1);
		q20 -= w19_q25_nb;
		q21 -= rotate_left(w19_q25_nb, 5);
	}
	WARP_TMP_BUF.flush2(Q27SOLBUF, Q27SOLCTL);
	PERF_STOP_COUNTER(25);
}

#ifdef LEGACY_Q36
__device__ void step_extend_Q36(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(36);
	using namespace dev;

	/// fetch the base solution, extended
	uint32_t  q27_sol0 = Q27SOLBUF.get<0>(thread_rd_idx);
	uint32_t  q27_sol1 = Q27SOLBUF.get<1>(thread_rd_idx);

	uint32_t  ext_idx 	= unpack_ext_idx(q27_sol0, q27_sol1);
	uint32_t  w19_sol_nb = unpack_w19_neutral_bits(q27_sol0, q27_sol1);
	uint32_t  w20_sol_nb = unpack_w20_neutral_bits(q27_sol0, q27_sol1);
	uint32_t  w21_sol_nb = unpack_w21_neutral_bits(q27_sol0, q27_sol1);

	uint32_t  base_idx = Q22SOLBUF.get<11>(ext_idx);
	uint32_t  q17 = Q22SOLBUF.get<0>(ext_idx);
	uint32_t  q18 = Q22SOLBUF.get<1>(ext_idx);
	uint32_t  q19 = Q22SOLBUF.get<2>(ext_idx);
	uint32_t  q20 = Q22SOLBUF.get<3>(ext_idx);
	uint32_t  q21 = Q22SOLBUF.get<4>(ext_idx);
	uint32_t  m6  = BASESOLBUF.get<6>(base_idx);
	uint32_t  m7  = BASESOLBUF.get<7>(base_idx);
	uint32_t  m8  = BASESOLBUF.get<8>(base_idx);
	uint32_t  m9  = BASESOLBUF.get<9>(base_idx);
	uint32_t  m10 = BASESOLBUF.get<10>(base_idx);
	uint32_t  m11 = BASESOLBUF.get<11>(base_idx);
	uint32_t  m12 = BASESOLBUF.get<12>(base_idx);
	uint32_t  m13 = BASESOLBUF.get<13>(base_idx);
	uint32_t  m19 = BASESOLBUF.get<19>(base_idx);
	uint32_t  m21 = BASESOLBUF.get<21>(base_idx);
	uint32_t  m14 = Q22SOLBUF.get<5>(ext_idx);
	uint32_t  m15 = Q22SOLBUF.get<6>(ext_idx);
	uint32_t  m16 = Q22SOLBUF.get<7>(ext_idx);
	uint32_t  m17 = Q22SOLBUF.get<8>(ext_idx);
	uint32_t  m18 = Q22SOLBUF.get<9>(ext_idx);
	uint32_t  m20 = Q22SOLBUF.get<10>(ext_idx);

	m19 += w19_sol_nb;
	m20 += w20_sol_nb;
	m21 += w21_sol_nb;
	uint32_t  m22 = sha1_mess(m19, m14, m8, m6);
	uint32_t  m23 = sha1_mess(m20, m15, m9, m7);
	uint32_t  m24 = sha1_mess(m21, m16, m10, m8);
	uint32_t  m25 = sha1_mess(m22, m17, m11, m9);
	uint32_t  m26 = sha1_mess(m23, m18, m12, m10);
	uint32_t  m27;
	uint32_t  m28;
	uint32_t  m29;
	uint32_t  m30;
	uint32_t  m31;
	uint32_t  m32;
	uint32_t  m33;
	uint32_t  m34;
	uint32_t  m35;

	q20 += w19_sol_nb;
	q21 += rotate_left(w19_sol_nb, 5) + w20_sol_nb;
	uint32_t  q22 = sha1_round2(q21, q20, q19, q18, q17, m21);
	uint32_t  q23 = sha1_round2(q22, q21, q20, q19, q18, m22);
	uint32_t  q24 = sha1_round2(q23, q22, q21, q20, q19, m23);
	uint32_t  q25 = sha1_round2(q24, q23, q22, q21, q20, m24);
	uint32_t  q26 = sha1_round2(q25, q24, q23, q22, q21, m25);
	uint32_t  q27 = sha1_round2(q26, q25, q24, q23, q22, m26);

	bool good36 = true;
	uint32_t  q32, q33, q34, q35, q36;

	do
	{
		m27 = sha1_mess(m24, m19, m13, m11);
		uint32_t  q28 = sha1_round2(q27, q26, q25, q24, q23, m27);
		uint32_t  nessies = Qset1mask[QOFF + 28]	/* ^ (Qprevmask  [QOFF + 28] & q27) */
												^ (Qprevrmask [QOFF + 28] & rotate_left(q27, 30))
												^ (Qprev2rmask[QOFF + 28] & rotate_left(q26, 30))
												;
		good36 = (0 == ((q28 ^ nessies) & Qcondmask[QOFF + 28]));
		if (!good36)
		{
			break;
		}
		m28 = sha1_mess(m25, m20, m14, m12);
		uint32_t  q29 = sha1_round2(q28, q27, q26, q25, q24, m28);
		nessies = /* Qset1mask[QOFF + 29]	^ (Qprevmask  [QOFF + 29] & q28) */
										/*^*/ (Qprevrmask [QOFF + 29] & rotate_left(q28, 30))
										/* ^ (Qprev2rmask[QOFF + 29] & rotate_left(q27, 30)) */
										;
		good36 = (0 == ((q29 ^ nessies) & Qcondmask[QOFF + 29]));
		if (!good36)
		{
			break;
		}
		m29 = sha1_mess(m26, m21, m15, m13);
		uint32_t  q30 = sha1_round2(q29, q28, q27, q26, q25, m29);
		nessies = Qset1mask[QOFF + 30]	/* ^ (Qprevmask  [QOFF + 30] & q29) */
										/* ^ (Qprevrmask [QOFF + 30] & rotate_left(q29, 30)) */
										/* ^ (Qprev2rmask[QOFF + 30] & rotate_left(q28, 30)) */
										;
		good36 = (0 == ((q30 ^ nessies) & Qcondmask[QOFF + 30]));
		if (!good36)
		{
			break;
		}
		m30 = sha1_mess(m27, m22, m16, m14);
		uint32_t  q31 = sha1_round2(q30, q29, q28, q27, q26, m30);
		nessies = Qset1mask[QOFF + 31]	/* ^ (Qprevmask  [QOFF + 31] & q30) */
										/* ^ (Qprevrmask [QOFF + 31] & rotate_left(q30, 30)) */
										^ (Qprev2rmask[QOFF + 31] & rotate_left(q29, 30))
										;
		good36 = (0 == ((q31 ^ nessies) & Qcondmask[QOFF + 31]));
		if (!good36)
		{
			break;
		}
		m31 = sha1_mess(m28, m23, m17, m15);
		q32 = sha1_round2(q31, q30, q29, q28, q27, m31);
		// no conditions at all!!
//		nessies = Qset1mask[QOFF + 32]	/* ^ (Qprevmask  [QOFF + 32] & q31) */
//										/* ^ (Qprevrmask [QOFF + 32] & rotate_left(q31, 30)) */
//										/* ^ (Qprev2rmask[QOFF + 32] & rotate_left(q30, 30)) */
//										;
//		good36 = (0 == ((q32 ^ nessies) & Qcondmask[QOFF + 32]));
//		if (!good36)
//		{
//			break;
//		}
		m32 = sha1_mess(m29, m24, m18, m16);
		q33 = sha1_round2(q32, q31, q30, q29, q28, m32);
		nessies = Qset1mask[QOFF + 33]	/* ^ (Qprevmask  [QOFF + 33] & q32) */
										^ (Qprevrmask [QOFF + 33] & rotate_left(q32, 30))
										/* ^ (Qprev2rmask[QOFF + 33] & rotate_left(q31, 30)) */
										;
		good36 = (0 == ((q33 ^ nessies) & Qcondmask[QOFF + 33]));
		if (!good36)
		{
			break;
		}
		m33 = sha1_mess(m30, m25, m19, m17);
		q34 = sha1_round2(q33, q32, q31, q30, q29, m33);
		m34 = sha1_mess(m31, m26, m20, m18);
		q35 = sha1_round2(q34, q33, q32, q31, q30, m34);
		m35 = sha1_mess(m32, m27, m21, m19);
		q36 = sha1_round2(q35, q34, q33, q32, q31, m35);

	} while (false);

	// sol: Q32,..,Q36,m20,...,m35
	Q36SOLBUF.write(Q36SOLCTL, good36, 	q32, q33, q34, q35, q36,
										m20, m21, m22, m23, m24, m25, m26, m27,
										m28, m29, m30, m31, m32, m33, m34, m35);
	PERF_STOP_COUNTER(36);
}

#else

__device__ void step_extend_Q36(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(36);
	using namespace dev;

	
	uint32_t  m20, m21, m22, m23;
	uint32_t  m24, m25, m26, m27;
	uint32_t  m28, m29, m30, m31;
	uint32_t  m32, m33, m34, m35;
	uint32_t  a, b, c, d, e;

	{
		/// fetch the base solution, extended
		uint32_t  base_idx;
		uint32_t  ext_idx;
		uint32_t  w19_sol_nb;
		uint32_t  w20_sol_nb;
		uint32_t  w21_sol_nb;
		{
			uint32_t  q27_sol0 = Q27SOLBUF.get<0>(thread_rd_idx);
			uint32_t  q27_sol1 = Q27SOLBUF.get<1>(thread_rd_idx);

			ext_idx	   = unpack_ext_idx(q27_sol0, q27_sol1);
			w19_sol_nb = unpack_w19_neutral_bits(q27_sol0, q27_sol1);
			w20_sol_nb = unpack_w20_neutral_bits(q27_sol0, q27_sol1);
			w21_sol_nb = unpack_w21_neutral_bits(q27_sol0, q27_sol1);
			base_idx = Q22SOLBUF.get<11>(ext_idx);
		}

		uint32_t  m6  = BASESOLBUF.get<6>(base_idx);
		uint32_t  m7  = BASESOLBUF.get<7>(base_idx);
		uint32_t  m8  = BASESOLBUF.get<8>(base_idx);
		uint32_t  m9  = BASESOLBUF.get<9>(base_idx);
		uint32_t  m10 = BASESOLBUF.get<10>(base_idx);
		uint32_t  m11 = BASESOLBUF.get<11>(base_idx);
		uint32_t  m12 = BASESOLBUF.get<12>(base_idx);
		uint32_t  m13 = BASESOLBUF.get<13>(base_idx);
		uint32_t  m19 = BASESOLBUF.get<19>(base_idx);

		uint32_t  m14 = Q22SOLBUF.get<5>(ext_idx);
		uint32_t  m15 = Q22SOLBUF.get<6>(ext_idx);
		uint32_t  m16 = Q22SOLBUF.get<7>(ext_idx);
		uint32_t  m17 = Q22SOLBUF.get<8>(ext_idx);
		uint32_t  m18 = Q22SOLBUF.get<9>(ext_idx);

		m21 = BASESOLBUF.get<21>(base_idx);
		m20 = Q22SOLBUF.get<10>(ext_idx);

		m19 += w19_sol_nb;
		m20 += w20_sol_nb;
		m21 += w21_sol_nb;

		m22 = sha1_mess(m19, m14, m8, m6);
		m23 = sha1_mess(m20, m15, m9, m7);
		m24 = sha1_mess(m21, m16, m10, m8);
		m25 = sha1_mess(m22, m17, m11, m9);
		m26 = sha1_mess(m23, m18, m12, m10);
		// there's some waste in computing these unconditionally
		// but it reduces register pressure
		m27 = sha1_mess(m24, m19, m13, m11);
		m28 = sha1_mess(m25, m20, m14, m12);
		m29 = sha1_mess(m26, m21, m15, m13);
		m30 = sha1_mess(m27, m22, m16, m14);
		m31 = sha1_mess(m28, m23, m17, m15);
		m32 = sha1_mess(m29, m24, m18, m16);
		m33 = sha1_mess(m30, m25, m19, m17);
		m34 = sha1_mess(m31, m26, m20, m18);
		m35 = sha1_mess(m32, m27, m21, m19);

		e = Q22SOLBUF.get<0>(ext_idx); // q17
		d = Q22SOLBUF.get<1>(ext_idx); // q18
		c = Q22SOLBUF.get<2>(ext_idx); // q19
		b = Q22SOLBUF.get<3>(ext_idx); // q20
		a = Q22SOLBUF.get<4>(ext_idx); // q21

		b += w19_sol_nb;
		a += rotate_left(w19_sol_nb, 5) + w20_sol_nb;

		e = sha1_round2(a, b, c, d, e, m21); // q22
		d = sha1_round2(e, a, b, c, d, m22); // q23
		c = sha1_round2(d, e, a, b, c, m23); // q24
		b = sha1_round2(c, d, e, a, b, m24); // q25
		a = sha1_round2(b, c, d, e, a, m25); // q26

		e = sha1_round2(a, b, c, d, e, m26); // q27
	}

	bool good36 = true;

#define QSET1M28 0x18000000
#define QSET1M30 0x80000000
#define QSET1M31 0x80000000
#define QSET1M33 0x20000000

#define QPREVR28 0x20000000
#define QPREVR29 0x18000000
#define QPREVR33 0x20000000

#define QPREV2R28 0x18000000
#define QPREV2R31 0x20000000

#define QCOND28 0x38000000
#define QCOND29 0x18000000
#define QCOND30 0x80000000
#define QCOND31 0xa0000000
#define QCOND33 0x20000000

	do
	{
		d = sha1_round2(e, a, b, c, d, m27); // q28
		uint32_t  nessies = QSET1M28 ^ (QPREVR28 & rotate_left(e, 30)) ^ (QPREV2R28 & rotate_left(a, 30));
		good36 = (0 == ((d ^ nessies) & QCOND28));
		if (!good36)
		{
			break;
		}
		c = sha1_round2(d, e, a, b, c, m28); // q29
		nessies = QPREVR29 & rotate_left(d, 30);
		good36 = (0 == ((c ^ nessies) & QCOND29));
		if (!good36)
		{
			break;
		}
		b = sha1_round2(c, d, e, a, b, m29); // q30
		nessies = QSET1M30;
		good36 = (0 == ((b ^ nessies) & QCOND30));
		if (!good36)
		{
			break;
		}
		a = sha1_round2(b, c, d, e, a, m30); // q31
		nessies = QSET1M31 ^ (QPREV2R31 & rotate_left(c, 30))
										;
		good36 = (0 == ((a ^ nessies) & QCOND31));
		if (!good36)
		{
			break;
		}
		e = sha1_round2(a, b, c, d, e, m31); // q32
		// no conditions at all!!

		d = sha1_round2(e, a, b, c, d, m32); // q33
		nessies = QSET1M33 ^ (QPREVR33 & rotate_left(e, 30));
		good36 = (0 == ((d ^ nessies) & QCOND33));
		if (!good36)
		{
			break;
		}
		c = sha1_round2(d, e, a, b, c, m33); // q34
		b = sha1_round2(c, d, e, a, b, m34); // q35
		a = sha1_round2(b, c, d, e, a, m35); // q36

	} while (false);

	// sol: Q32,..,Q36,m20,...,m35
	Q36SOLBUF.write(Q36SOLCTL, good36, 	e, d, c, b, a,
										m20, m21, m22, m23, m24, m25, m26, m27,
										m28, m29, m30, m31, m32, m33, m34, m35);
	PERF_STOP_COUNTER(36);
}
#endif

#ifndef NEW_Q56
// WARNING
// This is an old and deprecated version of the function, that saves good q32..q36
// It is no longer compatible with the way q56 solutions are tested for the full collision
__device__ void step_extend_Q56(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(56);
	using namespace dev;

	uint32_t  q32 = Q36SOLBUF.get<0>(thread_rd_idx);
	uint32_t  q33 = Q36SOLBUF.get<1>(thread_rd_idx);
	uint32_t  q34 = Q36SOLBUF.get<2>(thread_rd_idx);
	uint32_t  q35 = Q36SOLBUF.get<3>(thread_rd_idx);
	uint32_t  q36 = Q36SOLBUF.get<4>(thread_rd_idx);

	uint32_t  m20 = Q36SOLBUF.get<5>(thread_rd_idx);
	uint32_t  m21 = Q36SOLBUF.get<6>(thread_rd_idx);
	uint32_t  m22 = Q36SOLBUF.get<7>(thread_rd_idx);
	uint32_t  m23 = Q36SOLBUF.get<8>(thread_rd_idx);
	uint32_t  m24 = Q36SOLBUF.get<9>(thread_rd_idx);
	uint32_t  m25 = Q36SOLBUF.get<10>(thread_rd_idx);
	uint32_t  m26 = Q36SOLBUF.get<11>(thread_rd_idx);
	uint32_t  m27 = Q36SOLBUF.get<12>(thread_rd_idx);
	uint32_t  m28 = Q36SOLBUF.get<13>(thread_rd_idx);
	uint32_t  m29 = Q36SOLBUF.get<14>(thread_rd_idx);
	uint32_t  m30 = Q36SOLBUF.get<15>(thread_rd_idx);
	uint32_t  m31 = Q36SOLBUF.get<16>(thread_rd_idx);
	uint32_t  m32 = Q36SOLBUF.get<17>(thread_rd_idx);
	uint32_t  m33 = Q36SOLBUF.get<18>(thread_rd_idx);
	uint32_t  m34 = Q36SOLBUF.get<19>(thread_rd_idx);
	uint32_t  m35 = Q36SOLBUF.get<20>(thread_rd_idx);

	uint32_t  m36 = sha1_mess(m33, m28, m22, m20);
	uint32_t  m37 = sha1_mess(m34, m29, m23, m21);
	uint32_t  m38 = sha1_mess(m35, m30, m24, m22);
	uint32_t  m39 = sha1_mess(m36, m31, m25, m23);
	uint32_t  m40 = sha1_mess(m37, m32, m26, m24);
	uint32_t  m41 = sha1_mess(m38, m33, m27, m25);
	uint32_t  m42 = sha1_mess(m39, m34, m28, m26);
	uint32_t  m43 = sha1_mess(m40, m35, m29, m27);
	uint32_t  m44 = sha1_mess(m41, m36, m30, m28);
	uint32_t  m45 = sha1_mess(m42, m37, m31, m29);
	uint32_t  m46 = sha1_mess(m43, m38, m32, m30);
	uint32_t  m47 = sha1_mess(m44, m39, m33, m31);
	uint32_t  m48 = sha1_mess(m45, m40, m34, m32);
	uint32_t  m49 = sha1_mess(m46, m41, m35, m33);
	uint32_t  m50 = sha1_mess(m47, m42, m36, m34);
	uint32_t  m51 = sha1_mess(m48, m43, m37, m35);
	uint32_t  m52 = sha1_mess(m49, m44, m38, m36);
	uint32_t  m53 = sha1_mess(m50, m45, m39, m37);
	uint32_t  m54 = sha1_mess(m51, m46, m40, m38);
	uint32_t  m55 = sha1_mess(m52, m47, m41, m39);

	uint32_t  q37 = sha1_round2(q36, q35, q34, q33, q32, m36);
	uint32_t  q38 = sha1_round2(q37, q36, q35, q34, q33, m37);
	uint32_t  q39 = sha1_round2(q38, q37, q36, q35, q34, m38);
	uint32_t  q40 = sha1_round2(q39, q38, q37, q36, q35, m39);
	uint32_t  q41 = sha1_round3(q40, q39, q38, q37, q36, m40);
	uint32_t  q42 = sha1_round3(q41, q40, q39, q38, q37, m41);
	uint32_t  q43 = sha1_round3(q42, q41, q40, q39, q38, m42);
	uint32_t  q44 = sha1_round3(q43, q42, q41, q40, q39, m43);
	uint32_t  q45 = sha1_round3(q44, q43, q42, q41, q40, m44);
	uint32_t  q46 = sha1_round3(q45, q44, q43, q42, q41, m45);
	uint32_t  q47 = sha1_round3(q46, q45, q44, q43, q42, m46);
	uint32_t  q48 = sha1_round3(q47, q46, q45, q44, q43, m47);
	uint32_t  q49 = sha1_round3(q48, q47, q46, q45, q44, m48);
	uint32_t  q50 = sha1_round3(q49, q48, q47, q46, q45, m49);
	uint32_t  q51 = sha1_round3(q50, q49, q48, q47, q46, m50);
	uint32_t  q52 = sha1_round3(q51, q50, q49, q48, q47, m51);
	uint32_t  q53 = sha1_round3(q52, q51, q50, q49, q48, m52);
	uint32_t  q54 = sha1_round3(q53, q52, q51, q50, q49, m53);
	uint32_t  q55 = sha1_round3(q54, q53, q52, q51, q50, m54);
	uint32_t  q56 = sha1_round3(q55, q54, q53, q52, q51, m55);

	m36 ^= DV_DW[36];
	m37 ^= DV_DW[37];
	m38 ^= DV_DW[38];
	m39 ^= DV_DW[39];
	m40 ^= DV_DW[40];
	m41 ^= DV_DW[41];
	m42 ^= DV_DW[42];
	m43 ^= DV_DW[43];
	m44 ^= DV_DW[44];
	m45 ^= DV_DW[45];
	m46 ^= DV_DW[46];
	m47 ^= DV_DW[47];
	m48 ^= DV_DW[48];
	m49 ^= DV_DW[49];
	m50 ^= DV_DW[50];
	m51 ^= DV_DW[51];
	m52 ^= DV_DW[52];
	m53 ^= DV_DW[53];
	m54 ^= DV_DW[54];
	m55 ^= DV_DW[55];

	uint32_t  Q37 = sha1_round2(q36, q35, q34, q33, q32, m36);
	uint32_t  Q38 = sha1_round2(Q37, q36, q35, q34, q33, m37);
	uint32_t  Q39 = sha1_round2(Q38, Q37, q36, q35, q34, m38);
	uint32_t  Q40 = sha1_round2(Q39, Q38, Q37, q36, q35, m39);
	uint32_t  Q41 = sha1_round3(Q40, Q39, Q38, Q37, q36, m40);
	uint32_t  Q42 = sha1_round3(Q41, Q40, Q39, Q38, Q37, m41);
	uint32_t  Q43 = sha1_round3(Q42, Q41, Q40, Q39, Q38, m42);
	uint32_t  Q44 = sha1_round3(Q43, Q42, Q41, Q40, Q39, m43);
	uint32_t  Q45 = sha1_round3(Q44, Q43, Q42, Q41, Q40, m44);
	uint32_t  Q46 = sha1_round3(Q45, Q44, Q43, Q42, Q41, m45);
	uint32_t  Q47 = sha1_round3(Q46, Q45, Q44, Q43, Q42, m46);
	uint32_t  Q48 = sha1_round3(Q47, Q46, Q45, Q44, Q43, m47);
	uint32_t  Q49 = sha1_round3(Q48, Q47, Q46, Q45, Q44, m48);
	uint32_t  Q50 = sha1_round3(Q49, Q48, Q47, Q46, Q45, m49);
	uint32_t  Q51 = sha1_round3(Q50, Q49, Q48, Q47, Q46, m50);
	uint32_t  Q52 = sha1_round3(Q51, Q50, Q49, Q48, Q47, m51);
	uint32_t  Q53 = sha1_round3(Q52, Q51, Q50, Q49, Q48, m52);
	uint32_t  Q54 = sha1_round3(Q53, Q52, Q51, Q50, Q49, m53);
	uint32_t  Q55 = sha1_round3(Q54, Q53, Q52, Q51, Q50, m54);
	uint32_t  Q56 = sha1_round3(Q55, Q54, Q53, Q52, Q51, m55);

	bool good56 = (q52 == Q52);
	good56 = (q53 == Q53) && good56;
	good56 = (q54 == Q54) && good56;
	good56 = (q55 == Q55) && good56;
	good56 = (q56 == Q56) && good56;

	// sol: Q32,..,Q36,m6,...,m21
	COLLCANDIDATEBUF.write(COLLCANDIDATECTL, good56, q32, q33, q34, q35, q36,
											 m20, m21, m22, m23, m24, m25, m26, m27,
											 m28, m29, m30, m31, m32, m33, m34, m35);
	PERF_STOP_COUNTER(56);
}

#else

__device__ void step_extend_Q56(uint32_t  thread_rd_idx)
{
	PERF_START_COUNTER(56);
	using namespace dev;

	uint32_t  e = Q36SOLBUF.get<0>(thread_rd_idx);
	uint32_t  d = Q36SOLBUF.get<1>(thread_rd_idx);
	uint32_t  c = Q36SOLBUF.get<2>(thread_rd_idx);
	uint32_t  b = Q36SOLBUF.get<3>(thread_rd_idx);
	uint32_t  a = Q36SOLBUF.get<4>(thread_rd_idx);
	uint32_t  E = e;
	uint32_t  D = d;
	uint32_t  C = c;
	uint32_t  B = b;
	uint32_t  A = a;

	uint32_t  m36, m37, m38, m39, m40, m41, m42;
	uint32_t  m43, m44, m45, m46, m47, m48, m49;
	uint32_t  m50, m51, m52, m53, m54, m55;

	{
		uint32_t  m20 = Q36SOLBUF.get<5>(thread_rd_idx);
		uint32_t  m21 = Q36SOLBUF.get<6>(thread_rd_idx);
		uint32_t  m22 = Q36SOLBUF.get<7>(thread_rd_idx);
		uint32_t  m23 = Q36SOLBUF.get<8>(thread_rd_idx);
		uint32_t  m24 = Q36SOLBUF.get<9>(thread_rd_idx);
		uint32_t  m25 = Q36SOLBUF.get<10>(thread_rd_idx);
		uint32_t  m26 = Q36SOLBUF.get<11>(thread_rd_idx);
		uint32_t  m27 = Q36SOLBUF.get<12>(thread_rd_idx);
		uint32_t  m28 = Q36SOLBUF.get<13>(thread_rd_idx);
		uint32_t  m29 = Q36SOLBUF.get<14>(thread_rd_idx);
		uint32_t  m30 = Q36SOLBUF.get<15>(thread_rd_idx);
		uint32_t  m31 = Q36SOLBUF.get<16>(thread_rd_idx);
		uint32_t  m32 = Q36SOLBUF.get<17>(thread_rd_idx);
		uint32_t  m33 = Q36SOLBUF.get<18>(thread_rd_idx);
		uint32_t  m34 = Q36SOLBUF.get<19>(thread_rd_idx);
		uint32_t  m35 = Q36SOLBUF.get<20>(thread_rd_idx);

		m36 = sha1_mess(m33, m28, m22, m20);
		m37 = sha1_mess(m34, m29, m23, m21);
		m38 = sha1_mess(m35, m30, m24, m22);
		m39 = sha1_mess(m36, m31, m25, m23);
		m40 = sha1_mess(m37, m32, m26, m24);
		m41 = sha1_mess(m38, m33, m27, m25);
		m42 = sha1_mess(m39, m34, m28, m26);
		m43 = sha1_mess(m40, m35, m29, m27);
		m44 = sha1_mess(m41, m36, m30, m28);
		m45 = sha1_mess(m42, m37, m31, m29);
		m46 = sha1_mess(m43, m38, m32, m30);
		m47 = sha1_mess(m44, m39, m33, m31);
		m48 = sha1_mess(m45, m40, m34, m32);
		m49 = sha1_mess(m46, m41, m35, m33);
		m50 = sha1_mess(m47, m42, m36, m34);
		m51 = sha1_mess(m48, m43, m37, m35);
		m52 = sha1_mess(m49, m44, m38, m36);
		m53 = sha1_mess(m50, m45, m39, m37);
		m54 = sha1_mess(m51, m46, m40, m38);
		m55 = sha1_mess(m52, m47, m41, m39);
	}

	e = sha1_round2(a, b, c, d, e, m36);
	d = sha1_round2(e, a, b, c, d, m37);
	c = sha1_round2(d, e, a, b, c, m38);
	b = sha1_round2(c, d, e, a, b, m39);
	a = sha1_round3(b, c, d, e, a, m40);
	
	e = sha1_round3(a, b, c, d, e, m41);
	d = sha1_round3(e, a, b, c, d, m42);
	c = sha1_round3(d, e, a, b, c, m43);
	b = sha1_round3(c, d, e, a, b, m44);
	a = sha1_round3(b, c, d, e, a, m45);
	
	e = sha1_round3(a, b, c, d, e, m46);
	d = sha1_round3(e, a, b, c, d, m47);
	c = sha1_round3(d, e, a, b, c, m48);
	b = sha1_round3(c, d, e, a, b, m49);
	a = sha1_round3(b, c, d, e, a, m50);
	
	e = sha1_round3(a, b, c, d, e, m51);
	d = sha1_round3(e, a, b, c, d, m52);
	c = sha1_round3(d, e, a, b, c, m53);
	b = sha1_round3(c, d, e, a, b, m54);
	a = sha1_round3(b, c, d, e, a, m55);

	m36 ^= DV_DW[36];
	m37 ^= DV_DW[37];
	m38 ^= DV_DW[38];
	m39 ^= DV_DW[39];
	m40 ^= DV_DW[40];
	m41 ^= DV_DW[41];
	m42 ^= DV_DW[42];
	m43 ^= DV_DW[43];
	m44 ^= DV_DW[44];
	m45 ^= DV_DW[45];
	m46 ^= DV_DW[46];
	m47 ^= DV_DW[47];
	m48 ^= DV_DW[48];
	m49 ^= DV_DW[49];
	m50 ^= DV_DW[50];
	m51 ^= DV_DW[51];
	m52 ^= DV_DW[52];
	m53 ^= DV_DW[53];
	m54 ^= DV_DW[54];
	m55 ^= DV_DW[55];
	
	E = sha1_round2(A, B, C, D, E, m36);
	D = sha1_round2(E, A, B, C, D, m37);
	C = sha1_round2(D, E, A, B, C, m38);
	B = sha1_round2(C, D, E, A, B, m39);
	A = sha1_round3(B, C, D, E, A, m40);
	
	E = sha1_round3(A, B, C, D, E, m41);
	D = sha1_round3(E, A, B, C, D, m42);
	C = sha1_round3(D, E, A, B, C, m43);
	B = sha1_round3(C, D, E, A, B, m44);
	A = sha1_round3(B, C, D, E, A, m45);
	
	E = sha1_round3(A, B, C, D, E, m46);
	D = sha1_round3(E, A, B, C, D, m47);
	C = sha1_round3(D, E, A, B, C, m48);
	B = sha1_round3(C, D, E, A, B, m49);
	A = sha1_round3(B, C, D, E, A, m50);
	
	E = sha1_round3(A, B, C, D, E, m51);
	D = sha1_round3(E, A, B, C, D, m52);
	C = sha1_round3(D, E, A, B, C, m53);
	B = sha1_round3(C, D, E, A, B, m54);
	A = sha1_round3(B, C, D, E, A, m55);

	bool good56 = (e == E);
	good56 = (d == D) && good56;
	good56 = (c == C) && good56;
	good56 = (b == B) && good56;
	good56 = (a == A) && good56;

	// WARNING, DIFFERENT SEMANTICS
	// sol: Q52,..,Q56,m40,...,m55
	COLLCANDIDATEBUF.write(COLLCANDIDATECTL, good56, e, d, c, b, a,
											 m40, m41, m42, m43, m44, m45, m46, m47,
											 m48, m49, m50, m51, m52, m53, m54, m55);
	PERF_STOP_COUNTER(56);
}
#endif


// BACKUP CONTROLS ONLY IF THEY ARE IN SHARED (AND THUS BLOCK-SPECIFIC)
__device__ void backup_controls()
{
	__syncthreads();
	if (threadIdx.x == 0)
	{
		q18_solutions_ctl_bu[blockIdx.x] = Q18SOLCTL;
		q19_solutions_ctl_bu[blockIdx.x] = Q19SOLCTL;
		q20_solutions_ctl_bu[blockIdx.x] = Q20SOLCTL;
		q22_packed_solutions_ctl_bu[blockIdx.x] = Q22PACCTL;
		q22_solutions_ctl_bu[blockIdx.x] = Q22SOLCTL;
		q23_solutions_ctl_bu[blockIdx.x] = Q23SOLCTL;
		q24_solutions_ctl_bu[blockIdx.x] = Q24SOLCTL;
		q27_solutions_ctl_bu[blockIdx.x] = Q27SOLCTL;
		q36_solutions_ctl_bu[blockIdx.x] = Q36SOLCTL;
#ifdef USE_PERF_COUNTERS
		performance_backup();
#endif
	}
	__syncthreads();
}
__device__ void restore_controls()
{
	__syncthreads();
	if (threadIdx.x == 0)
	{
		Q18SOLCTL = q18_solutions_ctl_bu[blockIdx.x];
		Q19SOLCTL = q19_solutions_ctl_bu[blockIdx.x];
		Q20SOLCTL = q20_solutions_ctl_bu[blockIdx.x];
		Q22PACCTL = q22_packed_solutions_ctl_bu[blockIdx.x];
		Q22SOLCTL = q22_solutions_ctl_bu[blockIdx.x];
		Q23SOLCTL = q23_solutions_ctl_bu[blockIdx.x];
		Q24SOLCTL = q24_solutions_ctl_bu[blockIdx.x];
		Q27SOLCTL = q27_solutions_ctl_bu[blockIdx.x];
		Q36SOLCTL = q36_solutions_ctl_bu[blockIdx.x];
#ifdef USE_PERF_COUNTERS
		performance_restore();
#endif
	}
	__syncthreads();
}

__global__ void reset_buffers()
{
	// restore_controls(); // unnecessary

	BASESOLBUF.reset(BASESOLCTL);
	Q18SOLBUF.reset(Q18SOLCTL);
	Q19SOLBUF.reset(Q19SOLCTL);
	Q20SOLBUF.reset(Q20SOLCTL);
	Q22PACBUF.reset(Q22PACCTL);
	Q22SOLBUF.reset(Q22SOLCTL);
	Q23SOLBUF.reset(Q23SOLCTL);
	Q24SOLBUF.reset(Q24SOLCTL);
	Q27SOLBUF.reset(Q27SOLCTL);
	Q36SOLBUF.reset(Q36SOLCTL);
	COLLCANDIDATEBUF.reset(COLLCANDIDATECTL);

	backup_controls();
}

__global__ void cuda_attack()
{
	restore_controls();

	__shared__ uint64_t startclock;
	if (threadIdx.x==0)
	{
		startclock = clock64();
	}

#define USE_CLOCK_LOOP
#ifdef USE_CLOCK_LOOP
	do
	{
// 	crashes when adding this loop ?!?!?!?! causes memory access error ?!?!?
//	for (unsigned lloop = 0; lloop < 16; ++lloop) 
	{
#else
	for (unsigned lloop = 0; lloop < (1<<18); ++lloop)
	{
	{
#endif


#if 1
		{
			uint32_t thidx = Q36SOLBUF.getreadidx(Q36SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				step_extend_Q56(thidx);
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q27SOLBUF.getreadidx(Q27SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				step_extend_Q36(thidx);
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q24SOLBUF.getreadidx(Q24SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ256(thidx);
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q23SOLBUF.getreadidx(Q23SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ24(thidx);
				continue;
			}
		}
#endif

#if 1
		{
#ifdef PAC22
			uint32_t thidx = Q22PACBUF.getreadidx(Q22PACCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ23_aux(thidx);
				thidx = Q22SOLBUF.getreadidx(Q22SOLCTL);
				if (thidx != 0xFFFFFFFF)
				{
					stepQ23(thidx);
					continue;
				}
			}
#else
			uint32_t thidx = Q22SOLBUF.getreadidx(Q22SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ23(thidx);
				continue;
			}
#endif
		}
#endif

#if 1
		{
			uint32_t thidx = Q20SOLBUF.getreadidx(Q20SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ21(thidx);
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q19SOLBUF.getreadidx(Q19SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ20(thidx);
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q18SOLBUF.getreadidx(Q18SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ19(thidx);
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = BASESOLBUF.getreadidx(BASESOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ18(thidx);
				continue;
			}
		}
#endif

	}
	}
#ifdef USE_CLOCK_LOOP
	while ((clock64()-startclock) < (uint64_t(1)<<37));
#endif


	backup_controls();
}




void verify_step_computations(int cuda_blocks);
bool verify(basesol_t basesol);
void print_attack_info();


void save_q56solutions()
{
	if (outputfile.empty())
	{
		return;
	}
	static size_t oldsize = 0;

 	q56sol_t sol;
	vector<q56sol_t> q56sols;
	for (size_t i = 0; i < COLLCANDIDATECTL.write_idx; ++i)
	{
		sol.Q[0] = COLLCANDIDATEBUF.get<0>(i);
		sol.Q[1] = COLLCANDIDATEBUF.get<1>(i);
		sol.Q[2] = COLLCANDIDATEBUF.get<2>(i);
		sol.Q[3] = COLLCANDIDATEBUF.get<3>(i);
		sol.Q[4] = COLLCANDIDATEBUF.get<4>(i);

		sol.m[0] = COLLCANDIDATEBUF.get<5>(i);
		sol.m[1] = COLLCANDIDATEBUF.get<6>(i);
		sol.m[2] = COLLCANDIDATEBUF.get<7>(i);
		sol.m[3] = COLLCANDIDATEBUF.get<8>(i);
		sol.m[4] = COLLCANDIDATEBUF.get<9>(i);
		sol.m[5] = COLLCANDIDATEBUF.get<10>(i);
		sol.m[6] = COLLCANDIDATEBUF.get<11>(i);
		sol.m[7] = COLLCANDIDATEBUF.get<12>(i);
		sol.m[8] = COLLCANDIDATEBUF.get<13>(i);
		sol.m[9] = COLLCANDIDATEBUF.get<14>(i);
		sol.m[10] = COLLCANDIDATEBUF.get<15>(i);
		sol.m[11] = COLLCANDIDATEBUF.get<16>(i);
		sol.m[12] = COLLCANDIDATEBUF.get<17>(i);
		sol.m[13] = COLLCANDIDATEBUF.get<18>(i);
		sol.m[14] = COLLCANDIDATEBUF.get<19>(i);
		sol.m[15] = COLLCANDIDATEBUF.get<20>(i);
		q56sols.push_back(sol);
	}

	if (oldsize != q56sols.size())
	{
		oldsize = q56sols.size();
		cout << "Writing " << q56sols.size() << " Q56-solutions to '" << outputfile << "'..." << endl;

		ofstream ofs( (outputfile+".tmp").c_str(), ios::binary | ios::trunc);
		if (!ofs)
		{
			cout << "Error opening '" << outputfile << ".tmp'!" << endl;
			return;
		}
		ofs.write((char*)(&q56sols[0]),q56sols.size()*sizeof(q56sol_t));
		ofs.close();
		std::rename((outputfile+".tmp").c_str(), outputfile.c_str());
	}
}

buffer_basesol_t  basesol_buf_host;
control_basesol_t basesol_ctl_host;





bool compiled_with_cuda()
{
	return true;
}

void cuda_main(std::vector<basesol_t>& basesols)
{
//	print_attack_info();

	unsigned flags;
	switch (cuda_scheduler)
	{
		case 0: default:  flags = cudaDeviceScheduleAuto;         break;
		case 1:           flags = cudaDeviceScheduleSpin;         break;
		case 2:           flags = cudaDeviceScheduleYield;        break;
		case 3:           flags = cudaDeviceScheduleBlockingSync; break;
	}
	cout << "Using device " << cuda_device << ": " << flush;
	CUDA_ASSERT( cudaSetDeviceFlags(flags) );
	CUDA_ASSERT( cudaSetDevice(cuda_device) );
	cudaDeviceProp prop;
	CUDA_ASSERT( cudaGetDeviceProperties(&prop, cuda_device) );
	cout << prop.name << " (PCI " << hex << setw(2) << setfill('0') << prop.pciBusID << ":" << hex << setw(2) << setfill('0') << prop.pciDeviceID << "." << hex << prop.pciDomainID << dec << ")" << endl;

	if (cuda_threads_per_block == -1)
	{
		cuda_threads_per_block = prop.maxThreadsPerBlock;
	}
	if (THREADS_PER_BLOCK < cuda_threads_per_block)
	{
		cuda_threads_per_block = THREADS_PER_BLOCK;
	}
	if (cuda_blocks == -1)
	{
		cuda_blocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
	}
	if (BLOCKS < cuda_blocks)
	{
		cuda_blocks = BLOCKS;
	}
	cout << "Using " << cuda_blocks << " blocks x " << cuda_threads_per_block << " threads/block." << endl;


	cout << "Resetting buffers..." << flush;
	reset_buffers<<<cuda_blocks,32>>>();
	CUDA_ASSERT( cudaDeviceSynchronize() );
	cout << "done." << endl;

#ifdef USE_PERF_COUNTERS
	performance_reset();
#endif

#if 0
	cout << "Filtering base solutions" << endl;
	cout << "In : " << basesols.size() << endl;
	for (size_t i = 0; i < basesols.size(); )
	{
		if (verify(basesols[i]))
		{
			++i;
		} else
		{
			swap(basesols[i], basesols.back());
			basesols.pop_back();
		}
	}
	cout << "Out: " << basesols.size() << endl;
	if (basesols.size() == 0)
	{
		return;
	}
#endif

	cout << "Initializing base solution buffer" << endl;
	size_t basesolcnt = basesols.size();
	if (basesol_buf_host.size < basesolcnt)
	{
		basesolcnt = basesol_buf_host.size;
	}
	for (size_t i = 0; i < basesolcnt; ++i)
	{
		// Q12,..,Q17,m6,...,m21
		basesol_buf_host.write(basesol_ctl_host, true
			, basesols[i].Q[ 0], basesols[i].Q[ 1], basesols[i].Q[ 2], basesols[i].Q[ 3], basesols[i].Q[ 4], basesols[i].Q[ 5]
			, basesols[i].m[ 0], basesols[i].m[ 1], basesols[i].m[ 2], basesols[i].m[ 3]
			, basesols[i].m[ 4], basesols[i].m[ 5], basesols[i].m[ 6], basesols[i].m[ 7]
			, basesols[i].m[ 8], basesols[i].m[ 9], basesols[i].m[10], basesols[i].m[11]
			, basesols[i].m[12], basesols[i].m[13], basesols[i].m[14], basesols[i].m[15]
			);
	}
#ifdef USE_MANAGED
	cout << "Moving " << basesolcnt << " base solutions to GPU MANAGED..." << flush;
	// directly copy to variable in HOST memory
	base_solutions_buf = basesol_buf_host;
	base_solutions_ctl = basesol_ctl_host;
#else
	cout << "Moving " << basesolcnt << " base solutions to GPU GLOBAL..." << flush;
	// directly copy to variable in GPU GLOBAL memory
	CUDA_ASSERT( cudaMemcpyToSymbol(base_solutions_buf, &basesol_buf_host, sizeof(basesol_buf_host) ) );
	CUDA_ASSERT( cudaMemcpyToSymbol(base_solutions_ctl, &basesol_ctl_host, sizeof(basesol_ctl_host) ) );
	CUDA_ASSERT( cudaDeviceSynchronize() );
#endif
	cout << "done." << endl;

	cout << "Starting CUDA kernel" << flush;
	timer::timer cuda_total_time;
	while (true)
	{
		cout << "." << flush;

		timer::timer cuda_time;
		cuda_attack<<<cuda_blocks,cuda_threads_per_block>>>();
		CUDA_ASSERT( cudaDeviceSynchronize() );
		cout << "CUDA running time: " << cuda_time.time() << endl;

#ifdef VERIFY_GPU_RESULTS
		verify_step_computations(cuda_blocks);
#endif

		uint64_t q18sols = 0, q19sols = 0, q20sols = 0, q22sols = 0, q23sols = 0, q24sols = 0, q27sols = 0, q36sols = 0, q56sols = 0;
		for (unsigned bl = 0; bl < cuda_blocks; ++bl)
		{
			q18sols += q18_solutions_ctl_bu[bl].write_idx;
			q19sols += q19_solutions_ctl_bu[bl].write_idx;
			q20sols += q20_solutions_ctl_bu[bl].write_idx;
			q22sols += q22_solutions_ctl_bu[bl].write_idx;
			q23sols += q23_solutions_ctl_bu[bl].write_idx;
			q24sols += q24_solutions_ctl_bu[bl].write_idx;
			q27sols += q27_solutions_ctl_bu[bl].write_idx;
			q36sols += q36_solutions_ctl_bu[bl].write_idx;
		}
		q56sols = collision_candidates_ctl.write_idx;
		cout << "Q18 sols:\t" << q18sols << "\t" << (double(q18sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q19 sols:\t" << q19sols << "\t" << (double(q19sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q20 sols:\t" << q20sols << "\t" << (double(q20sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q22 sols:\t" << q22sols << "\t" << (double(q22sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q23 sols:\t" << q23sols << "\t" << (double(q23sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q24 sols:\t" << q24sols << "\t" << (double(q24sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q27 sols:\t" << q27sols << "\t" << (double(q27sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q36 sols:\t" << q36sols << "\t" << (double(q36sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q56 sols:\t" << q56sols << "\t" << (double(q56sols)/cuda_total_time.time()) << "#/s" << endl;

		save_q56solutions();

#ifdef USE_PERF_COUNTERS
		show_performance_counters();
#endif

		// exit if base solutions have been exhausted
		// !! NOTE THAT THERE MAY STILL BE SOME OTHER WORK LEFT !!
		uint32_t basesolsleft = base_solutions_ctl.write_idx - base_solutions_ctl.read_idx;
		cout << "Basesolutions left: " << basesolsleft << "\t" << (double(base_solutions_ctl.read_idx)/cuda_total_time.time()) << "#/s" << endl;
		if (basesolsleft < 32)
		{
			cout << "Exhausted base solutions!" << endl;
			break;
		}

	}
}




























/****************** STUFF TO DEBUG THE SOURCE CODE ************************/

string list_set_bits(uint32_t x)
{
	string ret;
	for (unsigned b = 0; b < 32; ++b)
	{
		if (x & (1<<b))
		{
			ret += " " + std::to_string(b);
		}
	}
	return ret;
}

//#define W14NBALLM 0x00003FE0
//#define W15NBALLM 0x00017FE0
//#define W16NBALLM 0x00012FC0
//#define W17NBALLM 0x00083C00
//#define W18NBALLM 0x00018000
//#define W19NBALLM 0x00005FC0
//#define W20NBALLM 0x00003841
//#define W21NBALLM 0x00030800
//#define W14NBQ18M 0x00003E00 // ..................xxxxx......... (5 of 'em)
//#define W15NBQ18M 0x00014000 // ...............x.x.............. (2 of 'em)
//#define W14NBQ19M 0x000001E0 // .......................xxxx..... (4 of 'em)
//#define W15NBQ19M 0x00003F00 // ..................xxxxxx........ (6 of 'em)
//#define W16NBQ19M 0x00012000 // ...............x..x............. (2 of 'em)
//#define W17NBQ19M 0x00080000 // ............x................... (1 of 'em)
//#define W15NBQ20M 0x000000E0 // ........................xxx..... (3 of 'em)
//#define W16NBQ20M 0x00000E00 // ....................xxx......... (3 of 'em)
//#define W16NBQ21M 0x000001C0 // .......................xxx...... (3 of 'em)
//#define W17NBQ21M 0x00003C00 // ..................xxxx.......... (4 of 'em)
//#define W18NBQ21M 0x00018000 // ...............xx............... (2 of 'em)
//#define W19NBQ23M 0x00005E00 // .................x.xxxx......... (5 of 'em)
//#define W19NBQ24M 0x000000C0 // ........................xx...... (2 of 'em)
//#define W20NBQ24M 0x00001800 // ...................xx........... (2 of 'em)
//#define W21NBQ24M 0x00030000 // ..............xx................ (2 of 'em)
//#define W19NBQ25M 0x00000100 // .......................x........ (1 of 'em)
//#define W20NBQ25M 0x00002040 // ..................x......x...... (2 of 'em)
//#define W21NBQ25M 0x00000800 // ....................x........... (1 of 'em)
//#define W20NBQ26M 0x00000001 // ...............................x (1 of 'em)

void print_attack_info()
{
	uint32_t w14m = 0, w15m = 0, w16m = 0, w17m = 0, w18m = 0, w19m = 0, w20m = 0, w21m = 0;

	cout << "Q18 neutral bits:" << endl;
	cout << "W14: " << list_set_bits(W14NBQ18M) << endl;
	cout << "W15: " << list_set_bits(W15NBQ18M) << endl;
	w14m ^= W14NBQ18M;
	w15m ^= W15NBQ18M;

	cout << "Q19 neutral bits:" << endl;
	cout << "W14: " << list_set_bits(W14NBQ19M) << endl;
	cout << "W15: " << list_set_bits(W15NBQ19M) << endl;
	cout << "W16: " << list_set_bits(W16NBQ19M) << endl;
	cout << "W17: " << list_set_bits(W17NBQ19M) << endl;
	w14m ^= W14NBQ19M;
	w15m ^= W15NBQ19M;
	w16m ^= W16NBQ19M;
	w17m ^= W17NBQ19M;

	cout << "Q20 neutral bits:" << endl;
	cout << "W15: " << list_set_bits(W15NBQ20M) << endl;
	cout << "W16: " << list_set_bits(W16NBQ20M) << endl;
	w15m ^= W15NBQ20M;
	w16m ^= W16NBQ20M;

	cout << "Q21 neutral bits:" << endl;
	cout << "W16: " << list_set_bits(W16NBQ21M) << endl;
	cout << "W17: " << list_set_bits(W17NBQ21M) << endl;
	cout << "W18: " << list_set_bits(W18NBQ21M) << endl;
	w16m ^= W16NBQ21M;
	w17m ^= W17NBQ21M;
	w18m ^= W18NBQ21M;

	cout << "Q23 neutral bits:" << endl;
	cout << "W19: " << list_set_bits(W19NBQ23M) << endl;
	w19m ^= W19NBQ23M;

	cout << "Q24 neutral bits:" << endl;
	cout << "W19: " << list_set_bits(W19NBQ24M) << endl;
	cout << "W20: " << list_set_bits(W20NBQ24M) << endl;
	cout << "W21: " << list_set_bits(W21NBQ24M) << endl;
	w19m ^= W19NBQ24M;
	w20m ^= W20NBQ24M;
	w21m ^= W21NBQ24M;

	cout << "Q25 neutral bits:" << endl;
	cout << "W19: " << list_set_bits(W19NBQ25M) << endl;
	cout << "W20: " << list_set_bits(W20NBQ25M) << endl;
	cout << "W21: " << list_set_bits(W21NBQ25M) << endl;
	w19m ^= W19NBQ25M;
	w20m ^= W20NBQ25M;
	w21m ^= W21NBQ25M;

	cout << "Q26 neutral bits:" << endl;
	cout << "W20: " << list_set_bits(W20NBQ26M) << endl;
	w20m ^= W20NBQ26M;

	cout << "All neutral bits:" << endl;
	cout << "W14: " << list_set_bits(W14NBALLM) << "\t\t err: " << list_set_bits(W14NBALLM ^ w14m) << endl;
	cout << "W15: " << list_set_bits(W15NBALLM) << "\t\t err: " << list_set_bits(W15NBALLM ^ w15m) << endl;
	cout << "W16: " << list_set_bits(W16NBALLM) << "\t\t err: " << list_set_bits(W16NBALLM ^ w16m) << endl;
	cout << "W17: " << list_set_bits(W17NBALLM) << "\t\t err: " << list_set_bits(W17NBALLM ^ w17m) << endl;
	cout << "W18: " << list_set_bits(W18NBALLM) << "\t\t err: " << list_set_bits(W18NBALLM ^ w18m) << endl;
	cout << "W19: " << list_set_bits(W19NBALLM) << "\t\t err: " << list_set_bits(W19NBALLM ^ w19m) << endl;
	cout << "W20: " << list_set_bits(W20NBALLM) << "\t\t err: " << list_set_bits(W20NBALLM ^ w20m) << endl;
	cout << "W21: " << list_set_bits(W21NBALLM) << "\t\t err: " << list_set_bits(W21NBALLM ^ w21m) << endl;
	cout << endl;
}






#ifdef VERIFY_GPU_RESULTS



#define VERIFY_ERROR(s) { cout << "Err @ block=" << block << " bufidx=" << read_idx << " baseidx=" << base_idx << " : " << s << endl; ok = false; }

bool isQokay(int t, uint32_t Q[])
{
	using namespace host;
	uint32_t Qval = Qset1mask[QOFF+t]
		^ (Qprevmask[QOFF+t] & Q[QOFF+t-1])
		^ (Qprevrmask[QOFF+t] & rotate_left(Q[QOFF+t-1],30))
		^ (Qprev2rmask[QOFF+t] & rotate_left(Q[QOFF+t-2],30))
		;
	return 0 == ( (Q[QOFF+t] ^ Qval) & Qcondmask[QOFF+t] );
}

bool verify_Q18_Q20(int block, size_t read_idx, int lastQ, uint32_t w0, uint32_t w1)
{
	bool ok = true;
	using namespace host;

	uint32_t m[80];
	uint32_t Q[85];

	size_t base_idx = unpack_base_idx(w0,w1);
	Q[QOFF+12] = base_solutions_buf.get<0>(base_idx);
	Q[QOFF+13] = base_solutions_buf.get<1>(base_idx);
	Q[QOFF+14] = base_solutions_buf.get<2>(base_idx);
	Q[QOFF+15] = base_solutions_buf.get<3>(base_idx);
	Q[QOFF+16] = base_solutions_buf.get<4>(base_idx);
	Q[QOFF+17] = base_solutions_buf.get<5>(base_idx);
	m[ 6] = base_solutions_buf.get< 6>(base_idx);
	m[ 7] = base_solutions_buf.get< 7>(base_idx);
	m[ 8] = base_solutions_buf.get< 8>(base_idx);
	m[ 9] = base_solutions_buf.get< 9>(base_idx);
	m[10] = base_solutions_buf.get<10>(base_idx);
	m[11] = base_solutions_buf.get<11>(base_idx);
	m[12] = base_solutions_buf.get<12>(base_idx);
	m[13] = base_solutions_buf.get<13>(base_idx);
	m[14] = base_solutions_buf.get<14>(base_idx);
	m[15] = base_solutions_buf.get<15>(base_idx);
	m[16] = base_solutions_buf.get<16>(base_idx);
	m[17] = base_solutions_buf.get<17>(base_idx);
	m[18] = base_solutions_buf.get<18>(base_idx);
	m[19] = base_solutions_buf.get<19>(base_idx);
	m[20] = base_solutions_buf.get<20>(base_idx);
	m[21] = base_solutions_buf.get<21>(base_idx);

	uint32_t m14nb = unpack_w14_neutral_bits(w0,w1);
	uint32_t m15nb = unpack_w15_neutral_bits(w0,w1);
	uint32_t m16nb = unpack_w16_neutral_bits(w0,w1);
	uint32_t m17nb = unpack_w17_neutral_bits(w0,w1);
	uint32_t m18nb = unpack_w18_neutral_bits(w0,w1);
	if (m14nb & ~W14NBALLM) VERIFY_ERROR("m14nb bad");
	if (m15nb & ~W15NBALLM) VERIFY_ERROR("m15nb bad");
	if (m16nb & ~W16NBALLM) VERIFY_ERROR("m16nb bad");
	if (m17nb & ~W17NBALLM) VERIFY_ERROR("m17nb bad");
	if (m18nb & ~W18NBALLM) VERIFY_ERROR("m18nb bad");

	m[14] |= m14nb;
	Q[QOFF+16] -= rotate_left(Q[QOFF+15],5);
	Q[QOFF+15] += m14nb;
	Q[QOFF+16] += rotate_left(Q[QOFF+15],5);

	m[15] |= m15nb;
	Q[QOFF+16] += m15nb;

	m[16] |= m16nb;
	m[17] |= m17nb;
	m[18] |= m18nb;
	for (int t = 16; t < lastQ; ++t)
	{
		sha1_step(t, Q, m);
	}

	for (int t = 15; t <= lastQ; ++t)
	{
		if (!isQokay(t,Q))
		{
			VERIFY_ERROR("Q" << t << " bad !");
		}
	}
	return ok;
}

bool verify_Q22(int block, size_t read_idx)
{
	bool ok = true;
	using namespace host;

	uint32_t m[80];
	uint32_t Q[85];

	uint32_t base_idx = q22_solutions_buf[block].get<11>(read_idx);
	Q[QOFF+12] = base_solutions_buf.get<0>(base_idx);
	Q[QOFF+13] = base_solutions_buf.get<1>(base_idx);
	Q[QOFF+14] = base_solutions_buf.get<2>(base_idx);
	Q[QOFF+15] = base_solutions_buf.get<3>(base_idx);
	Q[QOFF+16] = base_solutions_buf.get<4>(base_idx);
	Q[QOFF+17] = base_solutions_buf.get<5>(base_idx);
	m[ 6] = base_solutions_buf.get< 6>(base_idx);
	m[ 7] = base_solutions_buf.get< 7>(base_idx);
	m[ 8] = base_solutions_buf.get< 8>(base_idx);
	m[ 9] = base_solutions_buf.get< 9>(base_idx);
	m[10] = base_solutions_buf.get<10>(base_idx);
	m[11] = base_solutions_buf.get<11>(base_idx);
	m[12] = base_solutions_buf.get<12>(base_idx);
	m[13] = base_solutions_buf.get<13>(base_idx);
	m[14] = base_solutions_buf.get<14>(base_idx);
	m[15] = base_solutions_buf.get<15>(base_idx);
	m[16] = base_solutions_buf.get<16>(base_idx);
	m[17] = base_solutions_buf.get<17>(base_idx);
	m[18] = base_solutions_buf.get<18>(base_idx);
	m[19] = base_solutions_buf.get<19>(base_idx);
	m[20] = base_solutions_buf.get<20>(base_idx);
	m[21] = base_solutions_buf.get<21>(base_idx);

	uint32_t m14nb = q22_solutions_buf[block].get< 5>(read_idx) ^ m[14];
	uint32_t m15nb = q22_solutions_buf[block].get< 6>(read_idx) ^ m[15];
	uint32_t m16nb = q22_solutions_buf[block].get< 7>(read_idx) ^ m[16];
	uint32_t m17nb = q22_solutions_buf[block].get< 8>(read_idx) ^ m[17];
	uint32_t m18nb = q22_solutions_buf[block].get< 9>(read_idx) ^ m[18];
	if (m14nb & ~W14NBALLM) VERIFY_ERROR("m14nb bad");
	if (m15nb & ~W15NBALLM) VERIFY_ERROR("m15nb bad");
	if (m16nb & ~W16NBALLM) VERIFY_ERROR("m16nb bad");
	if (m17nb & ~W17NBALLM) VERIFY_ERROR("m17nb bad");
	if (m18nb & ~W18NBALLM) VERIFY_ERROR("m18nb bad");

	m[14] |= m14nb;
	Q[QOFF+16] -= rotate_left(Q[QOFF+15],5);
	Q[QOFF+15] += m14nb;
	Q[QOFF+16] += rotate_left(Q[QOFF+15],5);

	m[15] |= m15nb;
	Q[QOFF+16] += m15nb;

	m[16] |= m16nb;
	m[17] |= m17nb;
	m[18] |= m18nb;
	m[20] ^= ( ( (m15nb>>14) ^ (m15nb>>16) ^ (m16nb>>16) ^ (m18nb>>15) ) & 1) << 14;
	m[20] ^= ( ( (m15nb>>16) ) & 1) << 16;
	if (m[20] != q22_solutions_buf[block].get<10>(read_idx)) VERIFY_ERROR("Q22sol: m20 incorrect!: " << hex << (m[20] ^ q22_solutions_buf[block].get<10>(read_idx)));
	for (int t = 16; t < 22; ++t)
	{
		sha1_step(t, Q, m);
	}

	if (Q[QOFF+17] != q22_solutions_buf[block].get<0>(read_idx)) VERIFY_ERROR("Q22sol: Q17 incorrect!: " << hex << (Q[QOFF+17] ^ q22_solutions_buf[block].get<0>(read_idx)));
	if (Q[QOFF+18] != q22_solutions_buf[block].get<1>(read_idx)) VERIFY_ERROR("Q22sol: Q18 incorrect!: " << hex << (Q[QOFF+18] ^ q22_solutions_buf[block].get<1>(read_idx)));
	if (Q[QOFF+19] != q22_solutions_buf[block].get<2>(read_idx)) VERIFY_ERROR("Q22sol: Q19 incorrect!: " << hex << (Q[QOFF+19] ^ q22_solutions_buf[block].get<2>(read_idx)));
	if (Q[QOFF+20] != q22_solutions_buf[block].get<3>(read_idx)) VERIFY_ERROR("Q22sol: Q20 incorrect!: " << hex << (Q[QOFF+20] ^ q22_solutions_buf[block].get<3>(read_idx)));
	if (Q[QOFF+21] != q22_solutions_buf[block].get<4>(read_idx)) VERIFY_ERROR("Q22sol: Q21 incorrect!: " << hex << (Q[QOFF+21] ^ q22_solutions_buf[block].get<4>(read_idx)));

	for (int t = 15; t <= 22; ++t)
	{
		if (!isQokay(t,Q))
		{
			VERIFY_ERROR("Q22sol: Q" << t << " bad !");
		}
	}
	return ok;
}


bool verify_Q23_Q27(int block, size_t read_idx, int lastQ, uint32_t w0, uint32_t w1)
{
	bool ok = true;

	using namespace host;

	uint32_t m[80];
	uint32_t Q[85];

	size_t ext_base_idx = unpack_ext_idx(w0, w1);

	// load base solution
	size_t base_idx = q22_solutions_buf[block].get<11>(ext_base_idx);
	Q[QOFF+12] = base_solutions_buf.get<0>(base_idx);
	Q[QOFF+13] = base_solutions_buf.get<1>(base_idx);
	Q[QOFF+14] = base_solutions_buf.get<2>(base_idx);
	Q[QOFF+15] = base_solutions_buf.get<3>(base_idx);
	Q[QOFF+16] = base_solutions_buf.get<4>(base_idx);
	Q[QOFF+17] = base_solutions_buf.get<5>(base_idx);
	m[ 6] = base_solutions_buf.get< 6>(base_idx);
	m[ 7] = base_solutions_buf.get< 7>(base_idx);
	m[ 8] = base_solutions_buf.get< 8>(base_idx);
	m[ 9] = base_solutions_buf.get< 9>(base_idx);
	m[10] = base_solutions_buf.get<10>(base_idx);
	m[11] = base_solutions_buf.get<11>(base_idx);
	m[12] = base_solutions_buf.get<12>(base_idx);
	m[13] = base_solutions_buf.get<13>(base_idx);
	m[14] = base_solutions_buf.get<14>(base_idx);
	m[15] = base_solutions_buf.get<15>(base_idx);
	m[16] = base_solutions_buf.get<16>(base_idx);
	m[17] = base_solutions_buf.get<17>(base_idx);
	m[18] = base_solutions_buf.get<18>(base_idx);
	m[19] = base_solutions_buf.get<19>(base_idx);
	m[20] = base_solutions_buf.get<20>(base_idx);
	m[21] = base_solutions_buf.get<21>(base_idx);

	// overwrite Q15,..,Q21, m14,..,m18,m20 with extended base solution
	uint32_t m14nb = q22_solutions_buf[block].get< 5>(ext_base_idx) ^ m[14];
	uint32_t m15nb = q22_solutions_buf[block].get< 6>(ext_base_idx) ^ m[15];
	Q[QOFF+17] = q22_solutions_buf[block].get<0>(ext_base_idx);
	Q[QOFF+18] = q22_solutions_buf[block].get<1>(ext_base_idx);
	Q[QOFF+19] = q22_solutions_buf[block].get<2>(ext_base_idx);
	Q[QOFF+20] = q22_solutions_buf[block].get<3>(ext_base_idx);
	Q[QOFF+21] = q22_solutions_buf[block].get<4>(ext_base_idx);
	m[14]      = q22_solutions_buf[block].get<5>(ext_base_idx);
	m[15]      = q22_solutions_buf[block].get<6>(ext_base_idx);
	m[16]      = q22_solutions_buf[block].get<7>(ext_base_idx);
	m[17]      = q22_solutions_buf[block].get<8>(ext_base_idx);
	m[18]      = q22_solutions_buf[block].get<9>(ext_base_idx);
	m[20]      = q22_solutions_buf[block].get<10>(ext_base_idx);

	Q[QOFF+16] -= rotate_left(Q[QOFF+15],5);
	Q[QOFF+15] += m14nb;
	Q[QOFF+16] += rotate_left(Q[QOFF+15],5);
	Q[QOFF+16] += m15nb;

	// process neutral bits in w0 and w1
	uint32_t m19nb = unpack_w19_neutral_bits(w0, w1);
	uint32_t m20nb = unpack_w20_neutral_bits(w0, w1);
	uint32_t m21nb = unpack_w21_neutral_bits(w0, w1);
	if (m19nb & ~W19NBALLM) VERIFY_ERROR("Q" << lastQ << "sol: m19nb bad");
	if (m20nb & ~W20NBALLM) VERIFY_ERROR("Q" << lastQ << "sol: m20nb bad");
	if (m21nb & ~W21NBALLM) VERIFY_ERROR("Q" << lastQ << "sol: m21nb bad");

	m[19] ^= m19nb;
	m[20] ^= m20nb;
	m[21] ^= m21nb;

	for (int t = 22; t < lastQ; ++t)
	{
		m[t]=rotate_left(m[t-3] ^ m[t-8] ^ m[t-14] ^ m[t-16], 1);
	}

	for (int t = 19; t < lastQ; ++t)
	{
		sha1_step(t, Q, m);
	}

	for (int t = 19; t <= lastQ; ++t)
	{
		if (!isQokay(t,Q))
		{
			VERIFY_ERROR("Q" << lastQ << "sol: Q" << t << " bad !");
		}
	}
	return ok;
}



void verify_step_computations(int cuda_blocks)
{
	for (unsigned block = 0; block < cuda_blocks; ++block)
	{
		cout << "======== Verifying block " << block << endl;
		cout << "Base solutions left: " << (base_solutions_ctl.write_idx - base_solutions_ctl.read_idx) << setfill(' ') << endl;
		size_t q18checked = 0, q18ok = 0;
		size_t q18count = q18_solutions_ctl_bu[block].write_idx - q18_solutions_ctl_bu[block].read_idx;
		cout << q18_solutions_ctl_bu[block].read_idx << " " << q18_solutions_ctl_bu[block].write_idx << " " << q18count << endl;
		for (uint32_t i = q18_solutions_ctl_bu[block].read_idx; i != q18_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q18_solutions_buf[block].get<0>(i);
			uint32_t w1 = q18_solutions_buf[block].get<1>(i);
			if (verify_Q18_Q20(block, i, 18, w0, w1))
				++q18ok;
			++q18checked;
			if (i - q18_solutions_ctl_bu[block].read_idx > q18_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10) << q18checked << " out of " << setw(10) << q18count << " Q18 solutions: " << q18ok << " OK" << endl;

		size_t q19checked = 0, q19ok = 0;
		size_t q19count = q19_solutions_ctl_bu[block].write_idx - q19_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q19_solutions_ctl_bu[block].read_idx; i != q19_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q19_solutions_buf[block].get<0>(i);
			uint32_t w1 = q19_solutions_buf[block].get<1>(i);
			if (verify_Q18_Q20(block, i, 19, w0, w1))
				++q19ok;
			++q19checked;
			if (i - q19_solutions_ctl_bu[block].read_idx > q19_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q19checked << " out of " << setw(10) << q19count  << " Q19 solutions: " << q19ok << " OK" << endl;

		size_t q20checked = 0, q20ok = 0;
		size_t q20count = q20_solutions_ctl_bu[block].write_idx - q20_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q20_solutions_ctl_bu[block].read_idx; i != q20_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q20_solutions_buf[block].get<0>(i);
			uint32_t w1 = q20_solutions_buf[block].get<1>(i);
			if (verify_Q18_Q20(block, i, 20, w0, w1))
				++q20ok;
			++q20checked;
			if (i - q20_solutions_ctl_bu[block].read_idx > q20_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q20checked << " out of " << setw(10) << q20count  << " Q20 solutions: " << q20ok << " OK" << endl;

		size_t q22checked = 0, q22ok = 0;
		size_t q22count = q22_solutions_ctl_bu[block].write_idx - q22_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q22_solutions_ctl_bu[block].read_idx; i != q22_solutions_ctl_bu[block].write_idx; ++i)
		{
			if (verify_Q22(block, i))
				++q22ok;
			++q22checked;
			if (i - q22_solutions_ctl_bu[block].read_idx > q22_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q22checked << " out of " << setw(10) << q22count  << " Q22 solutions: " << q22ok << " OK" << endl;

		size_t q23checked = 0, q23ok = 0;
		size_t q23count = q23_solutions_ctl_bu[block].write_idx - q23_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q23_solutions_ctl_bu[block].read_idx; i != q23_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q23_solutions_buf[block].get<0>(i);
			uint32_t w1 = q23_solutions_buf[block].get<1>(i);
			if (verify_Q23_Q27(block, i, 23, w0, w1))
				++q23ok;
			++q23checked;
			if (i - q23_solutions_ctl_bu[block].read_idx > q23_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q23checked << " out of " << setw(10) << q23count  << " Q23 solutions: " << q23ok << " OK" << endl;

		size_t q24checked = 0, q24ok = 0;
		size_t q24count = q24_solutions_ctl_bu[block].write_idx - q24_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q24_solutions_ctl_bu[block].read_idx; i != q24_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q24_solutions_buf[block].get<0>(i);
			uint32_t w1 = q24_solutions_buf[block].get<1>(i);
			if (verify_Q23_Q27(block, i, 24, w0, w1))
				++q24ok;
			++q24checked;
			if (i - q24_solutions_ctl_bu[block].read_idx > q24_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q24checked << " out of " << setw(10) << q24count  << " Q24 solutions: " << q24ok << " OK" << endl;

		size_t q27checked = 0, q27ok = 0;
		size_t q27count = q27_solutions_ctl_bu[block].write_idx - q27_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q27_solutions_ctl_bu[block].read_idx; i != q27_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q27_solutions_buf[block].get<0>(i);
			uint32_t w1 = q27_solutions_buf[block].get<1>(i);
			if (verify_Q23_Q27(block, i, 27, w0, w1))
				++q27ok;
			++q27checked;
			if (i - q27_solutions_ctl_bu[block].read_idx > q27_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q27checked << " out of " << setw(10) << q27count  << " Q27 solutions: " << q27ok << " OK" << endl;

	}
}

#endif // VERIFY_GPU_RESULTS
