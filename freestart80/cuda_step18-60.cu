/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
            (C) 2016 Pierre Karpman

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

/************ TODO TODO **********\
\************ TODO TODO *********/


//// main prepocessor flags

// enables managed cyclic buffers and CPU verification of GPU results
//#define DEBUG1
// disabling temporary buffer will force writes to directly go to main buffer
#define DISABLE_TMP_BUF

// enable performance counters
//#define USE_PERF_COUNTERS
// PERFORMANCE COUNTERS ARE NOW WORKING PROPERLY AND HAVE VERY SMALL OVERHEAD


#ifndef DEBUG1
#define BLOCKS 26
#define THREADS_PER_BLOCK 512
#define DEBUG_BREAK
#else
#define BLOCKS 2
#define THREADS_PER_BLOCK 512
#define DEBUG_BREAK break;
#endif










#include "main.hpp"
#include "cyclicbuffer.hpp"
#include "neutral_bits_packing.hpp"

#include <map>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_ASSERT(s) 	{ cudaError_t err = s; if (err != cudaSuccess) { throw std::runtime_error("CUDA command returned: " + string(cudaGetErrorString(err)) + "!"); }  }




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

// definition of cyclic buffer for 2^16 22-word elems: basesol: Q12,..,Q17,m5,...,m20 [uses CAS, as it's only written by the host]
typedef cyclic_buffer_cas_t< BASESOLCOUNT, uint32_t, 22, cyclic_buffer_control_cas_t< BASESOLCOUNT > > buffer_basesol_t;
typedef buffer_basesol_t::control_t control_basesol_t;

// definition of cyclic buffer for 2^20 2-word elems
typedef cyclic_buffer_mask_t< (1<<20), uint32_t, 2, cyclic_buffer_control_mask_t< (1<<20) >, 1 > buffer_20_2_t; // used for block-specific buffers: fencetype = block-wide
typedef buffer_20_2_t::control_t control_20_2_t;

typedef cyclic_buffer_cas_t< (1<<20), uint32_t, 2, cyclic_buffer_control_cas_t< (1<<20) >, 2 > gl_buffer_20_2_t; // used for global buffers: fencetype = gpu-wide
typedef gl_buffer_20_2_t::control_t gl_control_20_2_t;

// definition of cyclic buffer for 2^20 11-word elems: extbasesol: q15, q16, q17, m14, m15, m16, m17, m18, m19, m20, base_idx
typedef cyclic_buffer_mask_t< (1<<20), uint32_t, 11, cyclic_buffer_control_mask_t< (1<<20) >, 1 > buffer_extbasesol20_t; // used for block-specific buffers: fencetype = block-wide
typedef buffer_extbasesol20_t::control_t control_extbasesol20_t;

// Marc: let's try steps 28-30 with 3 conditions total together and output extbasesol20+m17+m18+m19+booms
typedef cyclic_buffer_mask_t< (1 << 20), uint32_t, 3, cyclic_buffer_control_mask_t< (1 << 20) >, 1 > buffer_30_3_t; // used for block-specific buffers: fencetype = block-wide
typedef buffer_30_3_t::control_t control_30_3_t;

// definition of cyclic buffer for 2^10 21-word elems: sol: Q36,..,Q40,m24,...,m39
// definition of cyclic buffer for 2^10 21-word elems: sol: Q56,..,Q60,m44,...,m59
typedef cyclic_buffer_cas_t< (1<<10), uint32_t, 21, cyclic_buffer_control_cas_t< (1<<10) >, 2 > buffer_sol_t;
typedef buffer_sol_t::control_t control_sol_t;


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

// FIXME OLD
/*** Main buffers declaration ***/
MANAGED __device__ buffer_basesol_t  base_solutions_buf;
__managed__ __device__ control_basesol_t base_solutions_ctl;  // always managed to detect when it's empty
#define BASESOLBUF                   base_solutions_buf
#define BASESOLCTL                   base_solutions_ctl

MANAGED __device__ gl_buffer_20_2_t  q18_solutions_buf;
__managed__ __device__ gl_control_20_2_t q18_solutions_ctl;
#define Q18SOLBUF                 q18_solutions_buf
#define Q18SOLCTL                 q18_solutions_ctl

MANAGED __device__ gl_buffer_20_2_t  q19_solutions_buf;
__managed__ __device__ gl_control_20_2_t q19_solutions_ctl;
#define Q19SOLBUF                 q19_solutions_buf
#define Q19SOLCTL                 q19_solutions_ctl

MANAGED __device__ buffer_extbasesol20_t  q20_solutions_buf[BLOCKS];
__shared__ control_extbasesol20_t q20_solutions_ctl;
MANAGED2 __device__ control_extbasesol20_t q20_solutions_ctl_bu [BLOCKS];
#define Q20SOLBUF                       q20_solutions_buf    [blockIdx.x]
#define Q20SOLCTL                       q20_solutions_ctl

MANAGED __device__ buffer_20_2_t  q21_solutions_buf[BLOCKS];
__shared__ control_20_2_t q21_solutions_ctl;
MANAGED2 __device__ control_20_2_t q21_solutions_ctl_bu [BLOCKS];
#define Q21SOLBUF                 q21_solutions_buf    [blockIdx.x]
#define Q21SOLCTL                 q21_solutions_ctl

MANAGED __device__ buffer_20_2_t  q22_solutions_buf[BLOCKS];
__shared__ control_20_2_t q22_solutions_ctl;
MANAGED2 __device__ control_20_2_t q22_solutions_ctl_bu [BLOCKS];
#define Q22SOLBUF                 q22_solutions_buf    [blockIdx.x]
#define Q22SOLCTL                 q22_solutions_ctl

MANAGED __device__ buffer_20_2_t  q23_solutions_buf    [BLOCKS];
__shared__ control_20_2_t q23_solutions_ctl;
MANAGED2 __device__ control_20_2_t q23_solutions_ctl_bu [BLOCKS];
#define Q23SOLBUF                 q23_solutions_buf    [blockIdx.x]
#define Q23SOLCTL                 q23_solutions_ctl

MANAGED __device__ buffer_20_2_t  q26_solutions_buf    [BLOCKS];
__shared__ control_20_2_t q26_solutions_ctl;
MANAGED2 __device__ control_20_2_t q26_solutions_ctl_bu [BLOCKS];
#define Q26SOLBUF                 q26_solutions_buf    [blockIdx.x]
#define Q26SOLCTL                 q26_solutions_ctl

MANAGED __device__ buffer_30_3_t  q28_solutions_buf[BLOCKS];
__shared__ control_30_3_t q28_solutions_ctl;
MANAGED2 __device__ control_30_3_t q28_solutions_ctl_bu [BLOCKS];
#define Q28SOLBUF                 q28_solutions_buf    [blockIdx.x]
#define Q28SOLCTL                 q28_solutions_ctl


MANAGED __device__ buffer_30_3_t  q30_solutions_buf[BLOCKS];
__shared__ control_30_3_t q30_solutions_ctl;
MANAGED2 __device__ control_30_3_t q30_solutions_ctl_bu [BLOCKS];
#define Q30SOLBUF                 q30_solutions_buf    [blockIdx.x]
#define Q30SOLCTL                 q30_solutions_ctl

MANAGED __device__ buffer_sol_t  q40_solutions_buf[BLOCKS];
__shared__ control_sol_t q40_solutions_ctl;
MANAGED2 __device__ control_sol_t q40_solutions_ctl_bu [BLOCKS];
#define Q40SOLBUF                 q40_solutions_buf    [blockIdx.x]
#define Q40SOLCTL                 q40_solutions_ctl


__managed__ __device__ buffer_sol_t  collision_candidates_buf;
__managed__ __device__ control_sol_t collision_candidates_ctl;
#define COLLCANDIDATEBUF  collision_candidates_buf
#define COLLCANDIDATECTL  collision_candidates_ctl






/*** performance measure stuff ***/

#ifdef USE_PERF_COUNTERS

__managed__ uint64_t main_performance_counter[BLOCKS][80];
__shared__ uint32_t tmp_performance_counter[80];
__device__ inline void performance_reset()
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
				+= tmp_performance_counter[i];
		}
	}
}
__device__ inline void performance_restore()
{
	if (threadIdx.x == 0)
	{
		for (unsigned i = 0; i < 80; ++i)
		{
			//main_performance_counter[blockIdx.x][i]
			// =
			 tmp_performance_counter[i] = 0;
		}
	}
}
__device__ inline void performance_start_counter(unsigned i)
{
	if ((threadIdx.x&31)==0)
		atomicAdd(&tmp_performance_counter[i], (uint32_t)(uint64_t(0)-clock64()));
}
__device__ inline void performance_stop_counter(unsigned i)
{
	if ((threadIdx.x&31)==0)
		atomicAdd(&tmp_performance_counter[i], uint32_t(clock64()));
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


/* *** SHA1 FUNCTIONS **********************************
 */
__host__ __device__ inline uint32_t sha1_round1(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d, const uint32_t e, const uint32_t m)
{
//	a = rotate_left (a, 5);
//	c = rotate_left(c, 30);
//	d = rotate_left(d, 30);
//	e = rotate_left(e, 30);

	return rotate_left(a,5) + sha1_f1(b, rotate_left(c,30), rotate_left(d,30)) + rotate_left(e,30) + m + 0x5A827999;
}

__host__ __device__ inline uint32_t sha1_round2(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d, const uint32_t e, const uint32_t m)
//__host__ __device__ inline uint32_t sha1_round2(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e, uint32_t m)
{
//	a = rotate_left (a, 5);
//	c = rotate_left(c, 30);
//	d = rotate_left(d, 30);
//	e = rotate_left(e, 30);

//	return a + sha1_f2(b, c, d) + e + m + 0x6ED9EBA1;
	return rotate_left(a,5) + sha1_f2(b, rotate_left(c,30), rotate_left(d,30)) + rotate_left(e,30) + m + 0x6ED9EBA1;
}

__host__ __device__ inline uint32_t sha1_round3(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d, const uint32_t e, const uint32_t m)
{
//	a = rotate_left (a, 5);
//	c = rotate_left(c, 30);
//	d = rotate_left(d, 30);
//	e = rotate_left(e, 30);

	return rotate_left(a,5) + sha1_f3(b, rotate_left(c,30), rotate_left(d,30)) + rotate_left(e,30) + m + 0x8F1BBCDC;
//	return a + sha1_f3(b, c, d) + e + m + 0x8F1BBCDC;
}

__host__ __device__ inline uint32_t sha1_mess(uint32_t m_3, uint32_t m_8, uint32_t m_14, uint32_t m_16)
{
	return rotate_left(m_3 ^ m_8 ^ m_14 ^ m_16, 1);
}

#define NEXT_NB(a,mask) { (a) -= 1; (a) &= mask;}


__device__ void stepQ18(uint32_t base_idx)
{
	PERF_START_COUNTER(18);

	using namespace dev;

	/// fetch the base solution
	uint32_t q12 = BASESOLBUF.get<0>(base_idx);
	uint32_t q13 = BASESOLBUF.get<1>(base_idx);
	uint32_t q14 = BASESOLBUF.get<2>(base_idx);
	uint32_t q15 = BASESOLBUF.get<3>(base_idx);
	uint32_t q16 = BASESOLBUF.get<4>(base_idx);
	uint32_t q17 = BASESOLBUF.get<5>(base_idx);
	uint32_t m14 = BASESOLBUF.get<15>(base_idx);
	uint32_t m15 = BASESOLBUF.get<16>(base_idx);
	uint32_t m16 = BASESOLBUF.get<17>(base_idx);
	uint32_t m17 = BASESOLBUF.get<18>(base_idx);

	uint32_t oldq16 = q16;


	uint32_t w14_q18_nb = 0;
	for (unsigned l = 0; l < (1<<4); ++l)
	{
		NEXT_NB(w14_q18_nb, W14NBQ18M);

		m14 &= ~W14NBQ18M; 
		m14 |= w14_q18_nb;

		q15 += w14_q18_nb;
		q16 += rotate_left(w14_q18_nb, 5);

		uint32_t w15_q18_nb = 0;
		for (unsigned k = 0; k < (1<<4); ++k)
		{
			NEXT_NB(w15_q18_nb, W15NBQ18M);

			m15 &= ~W15NBQ18M;
			m15 |= w15_q18_nb;

			q16 += w15_q18_nb;

			bool valid_sol = (0 == ((oldq16 ^ q16) & Qcondmask[QOFF + 16]));

			uint32_t newq17 = sha1_round1(q16, q15, q14, q13, q12, m16);

			valid_sol &= 0 == ((newq17 ^ q17) & Qcondmask[QOFF + 17]);

			uint32_t newq18 = sha1_round1(newq17, q16, q15, q14, q13, m17);

			uint32_t q18nessies = Qset1mask[QOFF + 18] ^ (Qprevmask[QOFF + 18] & newq17)
				//					   											^ (Qprevrmask [QOFF + 18] & rotate_left(newq17, 30))
				//																^ (Qprev2rmask[QOFF + 18] & rotate_left(newq16, 30))
				;
			valid_sol &= 0 == ((newq18 ^ q18nessies) & Qcondmask[QOFF + 18]);

			uint32_t sol_val_0 = pack_q18q20_sol0(base_idx, m14, m15);
			uint32_t sol_val_1 = pack_q18q20_sol1(base_idx, m14, m15);
			WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q18SOLBUF, Q18SOLCTL);

			q16 -= w15_q18_nb;
		}

		q16 -= rotate_left(w14_q18_nb, 5);
		q15 -= w14_q18_nb; 
	}

	WARP_TMP_BUF.flush2(Q18SOLBUF, Q18SOLCTL);
	PERF_STOP_COUNTER(18);
}


__device__ void stepQ19(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(19);
	using namespace dev;

	uint32_t q18_sol0 = Q18SOLBUF.get<0>(thread_rd_idx);
	uint32_t q18_sol1 = Q18SOLBUF.get<1>(thread_rd_idx);

	uint32_t base_idx 	= unpack_idx(q18_sol0, q18_sol1);
	uint32_t w14_sol_nb = unpack_w14_nbs(q18_sol0, q18_sol1);
	uint32_t w15_sol_nb = unpack_w15_nbs(q18_sol0, q18_sol1);

	/// fetch the base solution and update it to the Q18 solution using the above neutral bits
	uint32_t q12 = BASESOLBUF.get<0>(base_idx);
	uint32_t q13 = BASESOLBUF.get<1>(base_idx);
	uint32_t q14 = BASESOLBUF.get<2>(base_idx);
	uint32_t q15 = BASESOLBUF.get<3>(base_idx);
	uint32_t q16 = BASESOLBUF.get<4>(base_idx);
	uint32_t m14 = BASESOLBUF.get<15>(base_idx);
	uint32_t m15 = BASESOLBUF.get<16>(base_idx);
	uint32_t m16 = BASESOLBUF.get<17>(base_idx);
	uint32_t m17 = BASESOLBUF.get<18>(base_idx);
	uint32_t m18 = BASESOLBUF.get<19>(base_idx);

	m14 |= w14_sol_nb;
	m15 |= w15_sol_nb;

	q15 += w14_sol_nb;
	q16 += w15_sol_nb + rotate_left(w14_sol_nb, 5);

	uint32_t oldq17  = sha1_round1(q16, q15, q14, q13, q12, m16); 
	uint32_t oldq18  = sha1_round1(oldq17, q16, q15, q14, q13, m17); 

	q16 -= rotate_left(q15, 5);

	uint32_t w14_q19_nb = 0;
	for (unsigned j = 0; j < (1<<2); ++j)
	{
		NEXT_NB(w14_q19_nb, W14NBQ19M);

		// start to recompute the previous state
		m14 &= ~W14NBQ19M;
		m14 |= w14_q19_nb;

		q15 += w14_q19_nb;
		q16 += rotate_left(q15, 5);

		uint32_t w15_q19_nb = 0;
		for (unsigned i = 0; i < (1<<5); ++i)
		{
			NEXT_NB(w15_q19_nb, W15NBQ19M);

			m15 &= ~W15NBQ19M;
			m15 |= w15_q19_nb;

			q16 += w15_q19_nb;

			uint32_t w16_q19_nb = 0;
			for (unsigned l = 0; l < (1<<5); ++l)
			{
				NEXT_NB(w16_q19_nb, W16NBQ19M);

				m16 &= ~W16NBQ19M;
				m16 |= w16_q19_nb;

				uint32_t q17  = sha1_round1(q16, q15, q14, q13, q12, m16); 
				uint32_t q18  = sha1_round1(q17, q16, q15, q14, q13, m17); 
				uint32_t q19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
				
				bool valid_sol = (0 == ((oldq17 ^ q17) & Qcondmask[QOFF + 17]));
				valid_sol &= 0 == ((oldq18 ^ q18) & Qcondmask[QOFF+18]);

				uint32_t q19nessies = Qset1mask[QOFF + 19]	^ (Qprevmask  [QOFF + 19] & q18)
															^ (Qprevrmask [QOFF + 19] & rotate_left(q18, 30))
															^ (Qprev2rmask[QOFF + 19] & rotate_left(q17, 30))
															;
				valid_sol &= 0 == ((q19 ^ q19nessies) & Qcondmask[QOFF + 19]);

				uint32_t sol_val_0 = pack_q18q20_sol0(base_idx, m14, m15, m16);
				uint32_t sol_val_1 = pack_q18q20_sol1(base_idx, m14, m15, m16);
				WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q19SOLBUF, Q19SOLCTL);
			}

			q16 -= w15_q19_nb;

		}
		q16 -= rotate_left(q15, 5);
		q15 -= w14_q19_nb;
	}

	WARP_TMP_BUF.flush2(Q19SOLBUF, Q19SOLCTL);
	PERF_STOP_COUNTER(19);
}


__device__ void stepQ20(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(20);
	using namespace dev;

	uint32_t q19_sol0 = Q19SOLBUF.get<0>(thread_rd_idx);
	uint32_t q19_sol1 = Q19SOLBUF.get<1>(thread_rd_idx);

	uint32_t base_idx 	= unpack_idx(q19_sol0, q19_sol1);
	uint32_t w14_sol_nb = unpack_w14_nbs(q19_sol0, q19_sol1);
	uint32_t w15_sol_nb = unpack_w15_nbs(q19_sol0, q19_sol1);
	uint32_t w16_sol_nb = unpack_w16_nbs(q19_sol0, q19_sol1);

	/// fetch the base solution and update it to the Q18 solution using the above neutral bits
	uint32_t q12 = BASESOLBUF.get<0>(base_idx);
	uint32_t q13 = BASESOLBUF.get<1>(base_idx);
	uint32_t q14 = BASESOLBUF.get<2>(base_idx);
	uint32_t q15 = BASESOLBUF.get<3>(base_idx);
	uint32_t q16 = BASESOLBUF.get<4>(base_idx);
	uint32_t m14 = BASESOLBUF.get<15>(base_idx);
	uint32_t m15 = BASESOLBUF.get<16>(base_idx);
	uint32_t m16 = BASESOLBUF.get<17>(base_idx);
	uint32_t m17 = BASESOLBUF.get<18>(base_idx);
	uint32_t m18 = BASESOLBUF.get<19>(base_idx);
	uint32_t m19 = BASESOLBUF.get<20>(base_idx);
	uint32_t m20 = BASESOLBUF.get<21>(base_idx);

	m14 |= w14_sol_nb;
	m15 |= w15_sol_nb;
	m16 |= w16_sol_nb;

	q15 += w14_sol_nb;
	q16 += w15_sol_nb + rotate_left(w14_sol_nb, 5);

	uint32_t oldq17  = sha1_round1(q16, q15, q14, q13, q12, m16); 
	uint32_t oldq18  = sha1_round1(oldq17, q16, q15, q14, q13, m17); 
	uint32_t oldq19  = sha1_round1(oldq18, oldq17, q16, q15, q14, m18); 

	q16 -= m15;

	uint32_t w15_q20_nb = 0;
	for (unsigned i = 0; i < (1<<2); ++i)
	{
		NEXT_NB(w15_q20_nb, W15NBQ20M);

		m15 &= ~W15NBQ20M;
		m15 |= w15_q20_nb;

		q16 += m15;

		uint32_t w16_q20_nb = 0;
		for (unsigned l = 0; l < (1<<4); ++l)
		{
			NEXT_NB(w16_q20_nb, W16NBQ20M);

			m16 &= ~W16NBQ20M;
			m16 |= w16_q20_nb;

			uint32_t q17  = sha1_round1(q16, q15, q14, q13, q12, m16); 

			uint32_t w17_q20_nb = 0;
			for (unsigned m = 0; m < (1<<6); ++m)
			{
				NEXT_NB(w17_q20_nb, W17NBQ20M);

				m17 &= ~W17NBQ20M;
				m17 |= w17_q20_nb;

				//w18[5]:  W15[4]
				uint32_t m18fb = (((m15 >> 4)) & 1) << 5;

				//w19[13]:  W15[15]  W15[16]  W17[14]  W17[16]  W17[19]
				uint32_t m19fb = ((0 ^ (m15 >> 15) ^ (m15 >> 16) ^ (m17 >> 14) ^ (m17 >> 16) ^ (m17 >> 19)) & 1) << 13;

				//w19[15] : W14[10]  W15[12]  W15[13]  W16[13]  W16[14]  W16[15]  W17[11]  W17[13]  W17[16]  W17[18]  W18[12]  W19[10]
				m19fb ^= ((0 ^ (m14 >> 10) ^ (m15 >> 12) ^ (m15 >> 13) ^ (m16 >> 13) ^ (m16 >> 14) ^ (m16 >> 15) ^ (m17 >> 11) ^ (m17 >> 13) ^ (m17 >> 16) ^ (m17 >> 18) ^ (m18 >> 12) ^ (m19 >> 10)) & 1) << 15;

				//w19[16] : W17[17]  W17[19]
				m19fb ^= ((0 ^ (m17 >> 17) ^ (m17 >> 19)) & 1) << 16;

				//w19[17] : W16[16]  W17[18]  W17[19]  W18[15]
				m19fb ^= ((0 ^ (m16 >> 16) ^ (m17 >> 18) ^ (m17 >> 19) ^ (m18 >> 15)) & 1) << 17;

				//w19[18] : W17[19]
				m19fb ^= ((0 ^ (m17 >> 19)) & 1) << 18;

				//w20[0] : W16[16]  W17[19]  W18[15]
				uint32_t m20fb = 0;
				m20fb ^= ((0 ^ (m16 >> 16) ^ (m17 >> 19) ^ (m18 >> 15)) & 1) << 0;

				//w20[14] : W15[14]  W15[16]  W17[18]  W17[19]
				m20fb ^= ((0 ^ (m15 >> 14) ^ (m15 >> 16) ^ (m17 >> 18) ^ (m17 >> 19)) & 1) << 14;

				//w20[16] : W15[16]
				m20fb ^= ((0 ^ (m15 >> 16)) & 1) << 16;

				//w20[17] : W15[16]  W16[13]  W16[15]  W17[16]  W18[12]
				m20fb ^= ((0 ^ (m15 >> 16) ^ (m16 >> 13) ^ (m16 >> 15) ^ (m17 >> 16) ^ (m18 >> 12)) & 1) << 17;

				m18 ^= m18fb;
				m19 ^= m19fb;
				m20 ^= m20fb;

				uint32_t q18  = sha1_round1(q17, q16, q15, q14, q13, m17); 
				uint32_t q19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
				uint32_t q20  = sha1_round1(q19, q18, q17, q16, q15, m19); 
				
				bool valid_sol = (0 == ((oldq17 ^ q17) & Qcondmask[QOFF + 17]));
				valid_sol &= 0 == ((oldq18 ^ q18) & Qcondmask[QOFF+18]);
				valid_sol &= 0 == ((oldq19 ^ q19) & Qcondmask[QOFF+19]);

				uint32_t q20nessies = Qset1mask[QOFF + 20]	^ (Qprevmask  [QOFF + 20] & q19)
															^ (Qprevrmask [QOFF + 20] & rotate_left(q19, 30))
															^ (Qprev2rmask[QOFF + 20] & rotate_left(q18, 30))
															;
				valid_sol &= 0 == ((q20 ^ q20nessies) & Qcondmask[QOFF + 20]);

				Q20SOLBUF.write(Q20SOLCTL, valid_sol, q15, q16, q17, m14, m15, m16, m17, m18, m19, m20, base_idx);

				m18 ^= m18fb;
				m19 ^= m19fb;
				m20 ^= m20fb;
			}

		}

		q16 -= m15;

	}
	PERF_STOP_COUNTER(20);
}



__device__ void stepQ21(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(21);
	using namespace dev;

	const uint32_t basesolidx = Q20SOLBUF.get<10>(thread_rd_idx);
	
	uint32_t q13 = BASESOLBUF.get<1>(basesolidx);
	uint32_t q14 = BASESOLBUF.get<2>(basesolidx);
	uint32_t q15 = Q20SOLBUF.get<0>(thread_rd_idx);
	uint32_t q16 = Q20SOLBUF.get<1>(thread_rd_idx);
	uint32_t q17 = Q20SOLBUF.get<2>(thread_rd_idx);

	uint32_t m17 = Q20SOLBUF.get<6>(thread_rd_idx);
	uint32_t m18 = Q20SOLBUF.get<7>(thread_rd_idx);
	uint32_t m19 = Q20SOLBUF.get<8>(thread_rd_idx);
	uint32_t m20 = Q20SOLBUF.get<9>(thread_rd_idx);

	uint32_t q18  = sha1_round1(q17, q16, q15, q14, q13, m17); 
	uint32_t oldq19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
	uint32_t oldq20  = sha1_round1(oldq19, q18, q17, q16, q15, m19); 
	q18 -= m17;

	uint32_t w17_q21_nb = 0;
	for (unsigned k = 0; k < (1<<4); ++k)
	{
		NEXT_NB(w17_q21_nb, W17NBQ21M);

		m17 &= ~W17NBQ21M;
		m17 |= w17_q21_nb;

		q18 += m17;
		
		uint32_t w18_q21_nb = 0;
		for (unsigned l = 0; l < (1<<1); ++l)
		{
			NEXT_NB(w18_q21_nb, W18NBQ21M);

			m18 &= ~W18NBQ21M;
			m18 |= w18_q21_nb;

			uint32_t m19fb = (((m17<<4)^(m17<<2))&(1<<15)) ^ ((m18<<2)&(1<<17));
			uint32_t m20fb = (m18>>15)&(1<<0);

			m19 ^= m19fb;
			m20 ^= m20fb;

			uint32_t q19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
			uint32_t q20  = sha1_round1(q19, q18, q17, q16, q15, m19); 
			uint32_t q21  = sha1_round2(q20, q19, q18, q17, q16, m20); 

			const uint32_t q21nessies = Qset1mask[QOFF+21]
//				^ (Qprevmask [QOFF + 21] & q20)
				^ (Qprevrmask [QOFF + 21] & rotate_left(q20, 30))
//				^ (Qprev2rmask [QOFF + 21] & rotate_left(q19, 30))
				;

			bool valid_sol = 0 == ((q21 ^ q21nessies) & Qcondmask[QOFF + 21]);
			valid_sol &= 0 == ((oldq20 ^ q20) & Qcondmask[QOFF+20]);
			valid_sol &= 0 == ((oldq19 ^ q19) & Qcondmask[QOFF+19]);

			uint32_t sol_val_0 = pack_q21q25_sol0(thread_rd_idx, m17, m18, m19, m20);
			uint32_t sol_val_1 = pack_q21q25_sol1(thread_rd_idx, m17, m18, m19, m20);
			WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q21SOLBUF, Q21SOLCTL);
			m19 ^= m19fb;
			m20 ^= m20fb;

		}

		q18 -= m17;
	}

	PERF_STOP_COUNTER(21);
}



__device__ void stepQ22(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(22);
	using namespace dev;

	uint32_t q21_sol0 = Q21SOLBUF.get<0>(thread_rd_idx);
	uint32_t q21_sol1 = Q21SOLBUF.get<1>(thread_rd_idx);

	const uint32_t extsolidx = unpack_idx(q21_sol0,q21_sol1);

	const uint32_t basesolidx = Q20SOLBUF.get<10>(extsolidx);
	
	uint32_t q13 = BASESOLBUF.get<1>(basesolidx);
	uint32_t q14 = BASESOLBUF.get<2>(basesolidx);
	uint32_t q15 = Q20SOLBUF.get<0>(extsolidx);
	uint32_t q16 = Q20SOLBUF.get<1>(extsolidx);
	uint32_t q17 = Q20SOLBUF.get<2>(extsolidx);

	uint32_t m13 = BASESOLBUF.get<14>(basesolidx);
	uint32_t m7 = BASESOLBUF.get<8>(basesolidx);
	uint32_t m5 = BASESOLBUF.get<6>(basesolidx);

	uint32_t m17 = Q20SOLBUF.get<6>(extsolidx);
	uint32_t m18 = Q20SOLBUF.get<7>(extsolidx);
	uint32_t m19 = Q20SOLBUF.get<8>(extsolidx);
	uint32_t m20 = Q20SOLBUF.get<9>(extsolidx);

	m17 ^= unpack_w17ext_nbs(q21_sol0,q21_sol1);
	m18 ^= unpack_w18_nbs(q21_sol0,q21_sol1);
	m19 &= ~W19NBPACKM;
	m19 ^= unpack_w19_nbs_fb(q21_sol0,q21_sol1);
	m20 &= ~W20NBPACKM;
	m20 ^= unpack_w20_nbs_fb(q21_sol0,q21_sol1);

	uint32_t q18  = sha1_round1(q17, q16, q15, q14, q13, m17); 
	uint32_t q19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
	uint32_t oldq20  = sha1_round1(q19, q18, q17, q16, q15, m19); 
//	uint32_t oldq21  = sha1_round2(oldq20, q19, q18, q17, q16, m20); 
	q19 -= m18;

	uint32_t m21precomp = m13 ^ m7 ^ m5;
	uint32_t q20precomp  = sha1_round1(0, q18, q17, q16, q15, 0); 

	uint32_t w18_q22_nb = 0;
	for (unsigned k = 0; k < (1<<6); k++)
	{
		NEXT_NB(w18_q22_nb, W18NBQ22M);

		m18 &= ~W18NBQ22M;
		m18 |= w18_q22_nb;


		q19 += m18;
		uint32_t m21 = rotate_left(m18 ^ m21precomp,1);

		uint32_t m19fb = ((m18<<3) /*^(m19<<5)*/) & (1<<15);
		uint32_t m20fb = (m18<<5) & (1<<17);

		m19 ^= m19fb;
		m20 ^= m20fb;

		uint32_t w19_q22_nb = 0;
		for (unsigned l = 0; l < (1<<2); l++)
		{
			NEXT_NB(w19_q22_nb, W19NBQ22M);

			m19 &= ~W19NBQ22M;
			m19 |= w19_q22_nb;
			m19 ^= (m19<<5) & (1<<15);

			uint32_t q20  = q20precomp + rotate_left(q19,5) + m19; //sha1_round1(q19, q18, q17, q16, q15, m19); 
			uint32_t q21  = sha1_round2(q20, q19, q18, q17, q16, m20); 
			uint32_t q22  = sha1_round2(q21, q20, q19, q18, q17, m21); 
			
			bool valid_sol = 0 == ((oldq20 ^ q20) & Qcondmask[QOFF+20]);

			const uint32_t q21nessies = Qset1mask[QOFF+21]
//				^ (Qprevmask [QOFF + 21] & newq20)
				^ (Qprevrmask [QOFF + 21] & rotate_left(q20, 30))
//				^ (Qprev2rmask [QOFF + 21] & rotate_left(q19, 30))
				;
			valid_sol &= 0 == ((q21 ^ q21nessies) & Qcondmask[QOFF + 21]);

			const uint32_t q22nessies = Qset1mask[QOFF+22]
				^ (Qprevmask [QOFF + 22] & q21)
//				^ (Qprevrmask [QOFF + 22] & rotate_left(q21, 30))
				^ (Qprev2rmask [QOFF + 22] & rotate_left(q20, 30))
				;
			valid_sol &= 0 == ((q22 ^ q22nessies) & Qcondmask[QOFF + 22]);

			uint32_t sol_val_0 = pack_q21q25_sol0(extsolidx, m17, m18, m19, m20);
			uint32_t sol_val_1 = pack_q21q25_sol1(extsolidx, m17, m18, m19, m20);
			WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q22SOLBUF, Q22SOLCTL);

			m19 ^= (m19<<5) & (1<<15);
		}
		m19 ^= m19fb;
		m20 ^= m20fb;

		q19 -= m18;
	}

	PERF_STOP_COUNTER(22);
}


__device__ void stepQ23(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(23);
	using namespace dev;

	uint32_t q22_sol0 = Q22SOLBUF.get<0>(thread_rd_idx);
	uint32_t q22_sol1 = Q22SOLBUF.get<1>(thread_rd_idx);

	const uint32_t extsolidx = unpack_idx(q22_sol0,q22_sol1);

	const uint32_t basesolidx = Q20SOLBUF.get<10>(extsolidx);
	
	uint32_t q13 = BASESOLBUF.get<1>(basesolidx);
	uint32_t q14 = BASESOLBUF.get<2>(basesolidx);
	uint32_t q15 = Q20SOLBUF.get<0>(extsolidx);
	uint32_t q16 = Q20SOLBUF.get<1>(extsolidx);
	uint32_t q17 = Q20SOLBUF.get<2>(extsolidx);

	uint32_t m5 = BASESOLBUF.get<6>(basesolidx);
	uint32_t m6 = BASESOLBUF.get<7>(basesolidx);
	uint32_t m7 = BASESOLBUF.get<8>(basesolidx);
	uint32_t m8 = BASESOLBUF.get<9>(basesolidx);
	uint32_t m13 = BASESOLBUF.get<14>(basesolidx);

	uint32_t m14 = Q20SOLBUF.get<3>(extsolidx);

	uint32_t m17 = Q20SOLBUF.get<6>(extsolidx);
	uint32_t m18 = Q20SOLBUF.get<7>(extsolidx);
	uint32_t m19 = Q20SOLBUF.get<8>(extsolidx);
	uint32_t m20 = Q20SOLBUF.get<9>(extsolidx);

	m17 ^= unpack_w17ext_nbs(q22_sol0,q22_sol1);
	m18 ^= unpack_w18_nbs(q22_sol0,q22_sol1);
	m19 &= ~W19NBPACKM;
	m19 ^= unpack_w19_nbs_fb(q22_sol0,q22_sol1);
	m20 &= ~W20NBPACKM;
	m20 ^= unpack_w20_nbs_fb(q22_sol0,q22_sol1);

	uint32_t q18  = sha1_round1(q17, q16, q15, q14, q13, m17); 
	uint32_t q19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
	uint32_t oldq20  = sha1_round1(q19, q18, q17, q16, q15, m19); 

	q19 -= m18;

	uint32_t m21precomp = m13 ^ m7 ^ m5;
	uint32_t m22precomp = m14 ^ m8 ^ m6;
	uint32_t q20precomp  = sha1_round1(0, q18, q17, q16, q15, 0); 

	uint32_t w18_q23_nb = 0;
	for (unsigned k = 0; k < (1<<4); k++)
	{
		NEXT_NB(w18_q23_nb, W18NBQ23M);

		m18 &= ~W18NBQ23M;
		m18 |= w18_q23_nb;

		q19 += m18;
		uint32_t m21 = rotate_left(m18 ^ m21precomp,1);

		uint32_t w19_q23_nb = 0;
		for (unsigned l = 0; l < (1<<3); l++)
		{
			NEXT_NB(w19_q23_nb, W19NBQ23M);

			m19 &= ~W19NBQ23M;
			m19 |= w19_q23_nb;

			uint32_t m22 = rotate_left(m19 ^ m22precomp,1);

			uint32_t q20  = q20precomp + rotate_left(q19,5) + m19; //sha1_round1(q19, q18, q17, q16, q15, m19); 

			uint32_t w20_q23_nb = 0;
			for (unsigned m = 0; m < (1<<1); m++)
			{
				NEXT_NB(w20_q23_nb, W20NBQ23M);

				m20 &= ~W20NBQ23M;
				m20 |= w20_q23_nb;

				uint32_t q21  = sha1_round2(q20, q19, q18, q17, q16, m20); 
				uint32_t q22  = sha1_round2(q21, q20, q19, q18, q17, m21); 
				uint32_t q23  = sha1_round2(q22, q21, q20, q19, q18, m22); 
			
				bool valid_sol = 0 == ((oldq20 ^ q20) & Qcondmask[QOFF+20]);

				const uint32_t q21nessies = Qset1mask[QOFF+21]
	//				^ (Qprevmask [QOFF + 21] & newq20)
					^ (Qprevrmask [QOFF + 21] & rotate_left(q20, 30))
	//				^ (Qprev2rmask [QOFF + 21] & rotate_left(q19, 30))
					;
				valid_sol &= 0 == ((q21 ^ q21nessies) & Qcondmask[QOFF + 21]);

				const uint32_t q22nessies = Qset1mask[QOFF+22]
					^ (Qprevmask [QOFF + 22] & q21)
	//				^ (Qprevrmask [QOFF + 22] & rotate_left(q21, 30))
					^ (Qprev2rmask [QOFF + 22] & rotate_left(q20, 30))
					;
				valid_sol &= 0 == ((q22 ^ q22nessies) & Qcondmask[QOFF + 22]);

				const uint32_t q23nessies = Qset1mask[QOFF+23]
					//					^ (Qprevmask [QOFF + 23] & q22)
					^ (Qprevrmask [QOFF + 23] & rotate_left(q22, 30))
					//					^ (Qprev2rmask [QOFF + 23] & rotate_left(q21, 30))
					;
				valid_sol &= 0 == ((q23 ^ q23nessies) & Qcondmask[QOFF + 23]);

				uint32_t sol_val_0 = pack_q21q25_sol0(extsolidx, m17, m18, m19, m20);
				uint32_t sol_val_1 = pack_q21q25_sol1(extsolidx, m17, m18, m19, m20);
				WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q23SOLBUF, Q23SOLCTL);
			}
		}

		q19 -= m18;
	}
	PERF_STOP_COUNTER(23);
}


__device__ void stepQ2456(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(24);
	using namespace dev;

	uint32_t q23_sol0 = Q23SOLBUF.get<0>(thread_rd_idx);
	uint32_t q23_sol1 = Q23SOLBUF.get<1>(thread_rd_idx);

	const uint32_t extsolidx = unpack_idx(q23_sol0,q23_sol1);

	const uint32_t basesolidx = Q20SOLBUF.get<10>(extsolidx);
	
	uint32_t q13 = BASESOLBUF.get<1>(basesolidx);
	uint32_t q14 = BASESOLBUF.get<2>(basesolidx);
	uint32_t q15 = Q20SOLBUF.get<0>(extsolidx);
	uint32_t q16 = Q20SOLBUF.get<1>(extsolidx);
	uint32_t q17 = Q20SOLBUF.get<2>(extsolidx);

	uint32_t m5 = BASESOLBUF.get<6>(basesolidx);
	uint32_t m6 = BASESOLBUF.get<7>(basesolidx);
	uint32_t m7 = BASESOLBUF.get<8>(basesolidx);
	uint32_t m8 = BASESOLBUF.get<9>(basesolidx);
	uint32_t m9 = BASESOLBUF.get<10>(basesolidx);
	uint32_t m10 = BASESOLBUF.get<11>(basesolidx);
	uint32_t m11 = BASESOLBUF.get<12>(basesolidx);
	uint32_t m13 = BASESOLBUF.get<14>(basesolidx);

	uint32_t m14 = Q20SOLBUF.get<3>(extsolidx);
	uint32_t m15 = Q20SOLBUF.get<4>(extsolidx);
	uint32_t m16 = Q20SOLBUF.get<5>(extsolidx);

	uint32_t m17 = Q20SOLBUF.get<6>(extsolidx);
	uint32_t m18 = Q20SOLBUF.get<7>(extsolidx);
	uint32_t m19 = Q20SOLBUF.get<8>(extsolidx);
	uint32_t m20 = Q20SOLBUF.get<9>(extsolidx);

	m17 ^= unpack_w17ext_nbs(q23_sol0,q23_sol1);
	m18 ^= unpack_w18_nbs(q23_sol0,q23_sol1);
	m19 &= ~W19NBPACKM;
	m19 ^= unpack_w19_nbs_fb(q23_sol0,q23_sol1);
	m20 &= ~W20NBPACKM;
	m20 ^= unpack_w20_nbs_fb(q23_sol0,q23_sol1);

	uint32_t m21 = rotate_left(m18 ^ m13 ^ m7 ^ m5,1);
	uint32_t m24 = rotate_left(m21 ^ m16 ^ m10 ^ m8,1);

	uint32_t q18  = sha1_round1(q17, q16, q15, q14, q13, m17); 
	uint32_t q19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
	uint32_t q20 = sha1_round1(q19, q18, q17, q16, q15, 0); 
	uint32_t q21precomp = sha1_round2(0, q19, q18, q17, q16, 0); 

	uint32_t m22precomp = m14 ^ m8 ^ m6;
	uint32_t m23precomp = m15 ^ m9 ^ m7;
	uint32_t m25precomp = m17 ^ m11 ^ m9;

	uint32_t w19_q24_nb = 0;
	for (unsigned l = 0; l < (1<<3); l++)
	{
		NEXT_NB(w19_q24_nb, W19NBQ24M);

		m19 &= ~W19NBQ24M;
		m19 |= w19_q24_nb;

		q20 += m19;

		uint32_t m22 = rotate_left(m19 ^ m22precomp,1);
		uint32_t m25 = rotate_left(m22 ^ m25precomp,1);

		uint32_t w20_q24_nb = 0;
#define W20NBQ2425M (W20NBQ24M|W20NBQ25M)
		for (unsigned m = 0; m < (1<<4); m++)
		{
			NEXT_NB(w20_q24_nb, W20NBQ2425M);

			m20 &= ~W20NBQ2425M;
			m20 |= w20_q24_nb;

			uint32_t m23 = rotate_left(m20 ^ m23precomp,1);

			uint32_t q21  = q21precomp + rotate_left(q20,5) + m20;
			uint32_t q22  = sha1_round2(q21, q20, q19, q18, q17, m21); 
			uint32_t q23  = sha1_round2(q22, q21, q20, q19, q18, m22); 
			uint32_t q24  = sha1_round2(q23, q22, q21, q20, q19, m23); 

			const uint32_t q22nessies = Qset1mask[QOFF+22]
				^ (Qprevmask [QOFF + 22] & q21)
//				^ (Qprevrmask [QOFF + 22] & rotate_left(q21, 30))
				^ (Qprev2rmask [QOFF + 22] & rotate_left(q20, 30))
				;
			bool valid_sol = 0 == ((q22 ^ q22nessies) & Qcondmask[QOFF + 22]);

			const uint32_t q23nessies = Qset1mask[QOFF+23]
				//					^ (Qprevmask [QOFF + 23] & q22)
									^ (Qprevrmask [QOFF + 23] & rotate_left(q22, 30))
				//					^ (Qprev2rmask [QOFF + 23] & rotate_left(q21, 30))
				;
			valid_sol &= 0 == ((q23 ^ q23nessies) & Qcondmask[QOFF + 23]);

			const uint32_t q24nessies = Qset1mask[QOFF+24]
				//					^ (Qprevmask [QOFF + 24] & q23)
									^ (Qprevrmask [QOFF + 24] & rotate_left(q23, 30))
									^ (Qprev2rmask [QOFF + 24] & rotate_left(q22, 30))
				;
			valid_sol &= 0 == ((q24 ^ q24nessies) & Qcondmask[QOFF + 24]);

			uint32_t q25  = sha1_round2(q24, q23, q22, q21, q20, m24); 
			const uint32_t q25nessies = Qset1mask[QOFF+25]
				//					^ (Qprevmask [QOFF + 25] & q24)
									^ (Qprevrmask [QOFF + 25] & rotate_left(q24, 30))
				//					^ (Qprev2rmask [QOFF + 25] & rotate_left(q23, 30))
				;
			valid_sol &= 0 == ((q25 ^ q25nessies) & Qcondmask[QOFF + 25]);

//			if (valid_sol)
//			{
			uint32_t q26  = sha1_round2(q25, q24, q23, q22, q21, m25); 
			const uint32_t q26nessies = Qset1mask[QOFF+26]
				//					^ (Qprevmask [QOFF + 26] & q25)
				//					^ (Qprevrmask [QOFF + 26] & rotate_left(q25, 30))
									^ (Qprev2rmask [QOFF + 26] & rotate_left(q24, 30))
				;
			valid_sol &= 0 == ((q26 ^ q26nessies) & Qcondmask[QOFF + 26]);
//			}

			uint32_t sol_val_0 = pack_q21q25_sol0(extsolidx, m17, m18, m19, m20);
			uint32_t sol_val_1 = pack_q21q25_sol1(extsolidx, m17, m18, m19, m20);
			WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q26SOLBUF, Q26SOLCTL);
		}
		q20 -= m19;
	}

	PERF_STOP_COUNTER(24);
}


__device__ void stepQ2728(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(27);
	using namespace dev;

	const uint32_t q26_sol0 = Q26SOLBUF.get<0>(thread_rd_idx);
	const uint32_t q26_sol1 = Q26SOLBUF.get<1>(thread_rd_idx);

	const uint32_t extsolidx = unpack_idx(q26_sol0,q26_sol1);
	const uint32_t basesolidx = Q20SOLBUF.get<10>(extsolidx);

	uint32_t m17 = Q20SOLBUF.get<6>(extsolidx);
	uint32_t m18 = Q20SOLBUF.get<7>(extsolidx);
	uint32_t m19 = Q20SOLBUF.get<8>(extsolidx);
	uint32_t m20 = Q20SOLBUF.get<9>(extsolidx);

	m17 ^= unpack_w17ext_nbs(q26_sol0,q26_sol1);
	m18 ^= unpack_w18_nbs(q26_sol0,q26_sol1);
	m19 &= ~W19NBPACKM;
	m19 ^= unpack_w19_nbs_fb(q26_sol0,q26_sol1);
	m20 &= ~W20NBPACKM;
	m20 ^= unpack_w20_nbs_fb(q26_sol0,q26_sol1);

	const uint32_t m5 = BASESOLBUF.get<6>(basesolidx);
	const uint32_t m7 = BASESOLBUF.get<8>(basesolidx);
	const uint32_t m13 = BASESOLBUF.get<14>(basesolidx);
	const uint32_t m21 = rotate_left(m18 ^ m13 ^ m7 ^ m5,1);

	uint32_t q19, q20, q21, q22, q23;
	{
		uint32_t q13 = BASESOLBUF.get<1>(basesolidx);
		uint32_t q14 = BASESOLBUF.get<2>(basesolidx);
		uint32_t q15 = Q20SOLBUF.get<0>(extsolidx);
		uint32_t q16 = Q20SOLBUF.get<1>(extsolidx);
		uint32_t q17 = Q20SOLBUF.get<2>(extsolidx);

		uint32_t q18  = sha1_round1(q17, q16, q15, q14, q13, m17); 
		q19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
		q20  = sha1_round1(q19, q18, q17, q16, q15, m19); 
		q21  = sha1_round2(q20, q19, q18, q17, q16, m20); 
		q22  = sha1_round2(q21, q20, q19, q18, q17, m21); 
		q23  = sha1_round2(q22, q21, q20, q19, q18, 0); 
	}

	// booms change: m10, m11, m12, m14, m15, m16 => m22 (m14), m23 (m15), m24 (m10,m16), m25(m11), m26(m10,m12),m27(m11),m28(m12,m14),m29(m15)

	// m5
	const uint32_t m6 = BASESOLBUF.get<7>(basesolidx);
	// m7
	const uint32_t m8 = BASESOLBUF.get<9>(basesolidx);
	const uint32_t m9 = BASESOLBUF.get<10>(basesolidx);
	uint32_t m10 = BASESOLBUF.get<11>(basesolidx);
	uint32_t m11 = BASESOLBUF.get<12>(basesolidx);
	uint32_t m12 = BASESOLBUF.get<13>(basesolidx);
	// m13

	uint32_t m14 = Q20SOLBUF.get<3>(extsolidx);
	uint32_t m15 = Q20SOLBUF.get<4>(extsolidx);
	uint32_t m16 = Q20SOLBUF.get<5>(extsolidx);

	uint32_t m22precomp = m19 ^ 0 ^ m8 ^ m6;
	uint32_t m23precomp = m20 ^ 0 ^ m9 ^ m7;
	uint32_t m24precomp = m21 ^ m16 ^ 0 ^ m8;
	uint32_t m25precomp = m17 ^ m9;
	uint32_t m26precomp = m18 ^ m12;
	uint32_t m27precomp = m19 ^ m13;
	uint32_t q24precomp  = sha1_round2(0, q22, q21, q20, q19, 0); 

	uint32_t q11_bo = 0;
	for (unsigned m = 0; m < (1<<2); ++m)
	{
		NEXT_NB(q11_bo, Q11BOOMS);

		m10 ^= q11_bo;
		m11 ^= q11_bo<<5;
		m14 ^= (q11_bo>>2)&(1<<6);
		m15 ^= q11_bo>>2;

		uint32_t m22 = rotate_left(m14 ^ m22precomp,1);
		q23 += m22;

		uint32_t m23 = rotate_left(m15 ^ m23precomp,1);
		uint32_t q24  = q24precomp + rotate_left(q23,5) + m23; //sha1_round2(q23, q22, q21, q20, q19, m23); 

		uint32_t m24 = rotate_left(m10 ^ m24precomp,1);
		uint32_t q25  = sha1_round2(q24, q23, q22, q21, q20, m24); 

		uint32_t m25 = rotate_left(m22 ^ m11 ^ m25precomp,1);
		uint32_t q26  = sha1_round2(q25, q24, q23, q22, q21, m25); 

		uint32_t m26 = rotate_left(m23 ^ m26precomp ^ m10,1);
		uint32_t q27  = sha1_round2(q26, q25, q24, q23, q22, m26); 

		uint32_t m27 = rotate_left(m24 ^ m27precomp ^ m11,1);
		uint32_t q28  = sha1_round2(q27, q26, q25, q24, q23, m27);

//		const uint32_t q25nessies = Qset1mask[QOFF+25]
			//					^ (Qprevmask [QOFF + 25] & q24)
//			^ (Qprevrmask [QOFF + 25] & rotate_left(q24,30))
			//					^ (Qprev2rmask [QOFF + 25] & rotate_left(q23, 30))
			;
//		bool valid_sol = 0 == ((q25 ^ q25nessies) & Qcondmask[QOFF + 25]);

		const uint32_t q26nessies = Qset1mask[QOFF+26]
			//					^ (Qprevmask [QOFF + 26] & q25)
			//					^ (Qprevrmask [QOFF + 26] & rotate_left(q25, 30))
			^ (Qprev2rmask [QOFF + 26] & rotate_left(q24,30))
			;
		bool valid_sol = 0 == ((q26 ^ q26nessies) & Qcondmask[QOFF + 26]);

		const uint32_t q27nessies = Qset1mask[QOFF+27]
			//					^ (Qprevmask [QOFF + 27] & q26)
			^ (Qprevrmask [QOFF + 27] & rotate_left(q26,30))
			//					^ (Qprev2rmask [QOFF + 27] & rotate_left(q25, 30))
			;
		valid_sol &= 0 == ((q27 ^ q27nessies) & Qcondmask[QOFF + 27]);

		const uint32_t q28nessies = // Qset1mask[QOFF+28]
			//					^ (Qprevmask [QOFF + 28] & q27)
			//					^ (Qprevrmask [QOFF + 28] & rotate_left(q27, 30))
			/*^*/ (Qprev2rmask [QOFF + 28] & rotate_left(q26,30))
			;
		valid_sol &= 0 == ((q28 ^ q28nessies) & Qcondmask[QOFF + 28]);

		Q28SOLBUF.write(Q28SOLCTL, valid_sol, q26_sol0, q26_sol1, q11_bo);

		q23 -= m22;
		m10 ^= q11_bo;
		m11 ^= q11_bo<<5;
		m14 ^= (q11_bo>>2)&(1<<6);
		m15 ^= q11_bo>>2;
	}
	PERF_STOP_COUNTER(27);
}

__device__ void stepQ2930(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(29);
	using namespace dev;

	const uint32_t q28_sol0 = Q28SOLBUF.get<0>(thread_rd_idx);
	const uint32_t q28_sol1 = Q28SOLBUF.get<1>(thread_rd_idx);
	const uint32_t q28_sol2 = Q28SOLBUF.get<2>(thread_rd_idx);

	const uint32_t extsolidx = unpack_idx(q28_sol0,q28_sol1);
	const uint32_t basesolidx = Q20SOLBUF.get<10>(extsolidx);

	uint32_t m17 = Q20SOLBUF.get<6>(extsolidx);
	uint32_t m18 = Q20SOLBUF.get<7>(extsolidx);
	uint32_t m19 = Q20SOLBUF.get<8>(extsolidx);
	uint32_t m20 = Q20SOLBUF.get<9>(extsolidx);

	m17 ^= unpack_w17ext_nbs(q28_sol0,q28_sol1);
	m18 ^= unpack_w18_nbs(q28_sol0,q28_sol1);
	m19 &= ~W19NBPACKM;
	m19 ^= unpack_w19_nbs_fb(q28_sol0,q28_sol1);
	m20 &= ~W20NBPACKM;
	m20 ^= unpack_w20_nbs_fb(q28_sol0,q28_sol1);

	const uint32_t m5 = BASESOLBUF.get<6>(basesolidx);
	const uint32_t m7 = BASESOLBUF.get<8>(basesolidx);
	const uint32_t m13 = BASESOLBUF.get<14>(basesolidx);
	const uint32_t m21 = rotate_left(m18 ^ m13 ^ m7 ^ m5,1);

	uint32_t q19, q20, q21, q22, q23;
	{
		uint32_t q13 = BASESOLBUF.get<1>(basesolidx);
		uint32_t q14 = BASESOLBUF.get<2>(basesolidx);
		uint32_t q15 = Q20SOLBUF.get<0>(extsolidx);
		uint32_t q16 = Q20SOLBUF.get<1>(extsolidx);
		uint32_t q17 = Q20SOLBUF.get<2>(extsolidx);

		uint32_t q18  = sha1_round1(q17, q16, q15, q14, q13, m17); 
		q19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
		q20  = sha1_round1(q19, q18, q17, q16, q15, m19); 
		q21  = sha1_round2(q20, q19, q18, q17, q16, m20); 
		q22  = sha1_round2(q21, q20, q19, q18, q17, m21); 
		q23  = sha1_round2(q22, q21, q20, q19, q18, 0); 
	}

	// booms change: m10, m11, m12, m14, m15, m16 => m22 (m14), m23 (m15), m24 (m10,m16), m25(m11), m26(m10,m12),m27(m11),m28(m12,m14),m29(m15)

	// m5
	const uint32_t m6 = BASESOLBUF.get<7>(basesolidx);
	// m7
	const uint32_t m8 = BASESOLBUF.get<9>(basesolidx);
	const uint32_t m9 = BASESOLBUF.get<10>(basesolidx);
	uint32_t m10 = BASESOLBUF.get<11>(basesolidx);
	uint32_t m11 = BASESOLBUF.get<12>(basesolidx);
	uint32_t m12 = BASESOLBUF.get<13>(basesolidx);
	// m13

	uint32_t m14 = Q20SOLBUF.get<3>(extsolidx);
	uint32_t m15 = Q20SOLBUF.get<4>(extsolidx);
	uint32_t m16 = Q20SOLBUF.get<5>(extsolidx);

	m10 ^= q28_sol2;
	m11 ^= q28_sol2<<5;
	m14 ^= (q28_sol2>>2)&(1<<6);
	m15 ^= q28_sol2>>2;

	uint32_t m22 = rotate_left(m19 ^ m14 ^ m8 ^ m6,1);
	q23 += m22;

	uint32_t m23 = rotate_left(m20 ^ m15 ^ m9 ^ m7,1);
	uint32_t q24  = sha1_round2(q23, q22, q21, q20, q19, m23); 
	uint32_t q25  = sha1_round2(q24, q23, q22, q21, q20, 0); 
	uint32_t q26precomp = sha1_round2(0, q24, q23, q22, q21, 0);

	uint32_t m24precomp = m21 ^ m10 ^ m8;
	uint32_t m25precomp = m22 ^ m17 ^ m9;
	uint32_t m26precomp = m23 ^ m18 ^ m10;
	uint32_t m27precomp = m19 ^ m13;
	uint32_t m29precomp = m21 ^ m15 ^ m13;

	uint32_t q12_bo = 0;
	for (unsigned m = 0; m < (1<<2); ++m)
	{
		NEXT_NB(q12_bo, Q12BOOMS);

		m11 ^= q12_bo;
		m12 ^= q12_bo<<5;
		m16 ^= q12_bo>>2;

		uint32_t m24 = rotate_left(m16 ^ m24precomp,1);
		q25 += m24;

		uint32_t m25 = rotate_left(m11 ^ m25precomp,1);
		uint32_t q26 = q26precomp + rotate_left(q25,5) + m25;
//		uint32_t q26  = sha1_round2(q25, q24, q23, q22, q21, m25); 

		uint32_t m26 = rotate_left(m12 ^ m26precomp,1);
		uint32_t q27  = sha1_round2(q26, q25, q24, q23, q22, m26); 

		uint32_t m27 = rotate_left(m24 ^ m11 ^ m27precomp,1);
		uint32_t q28  = sha1_round2(q27, q26, q25, q24, q23, m27);

		uint32_t m28 = rotate_left(m25 ^ m20 ^ m14 ^ m12,1);
		uint32_t q29  = sha1_round2(q28, q27, q26, q25, q24, m28);

		uint32_t m29 = rotate_left(m26 ^ m29precomp,1);
		uint32_t q30  = sha1_round2(q29, q28, q27, q26, q25, m29);

//		const uint32_t q25nessies = Qset1mask[QOFF+25]
			//					^ (Qprevmask [QOFF + 25] & q24)
//			^ (Qprevrmask [QOFF + 25] & rotate_left(q24,30))
			//					^ (Qprev2rmask [QOFF + 25] & rotate_left(q23, 30))
//			;
//		bool valid_sol = 0 == ((q25 ^ q25nessies) & Qcondmask[QOFF + 25]);

//		const uint32_t q26nessies = Qset1mask[QOFF+26]
			//					^ (Qprevmask [QOFF + 26] & q25)
			//					^ (Qprevrmask [QOFF + 26] & rotate_left(q25, 30))
//			^ (Qprev2rmask [QOFF + 26] & rotate_left(q24,30))
//			;
//		valid_sol &= 0 == ((q26 ^ q26nessies) & Qcondmask[QOFF + 26]);

		const uint32_t q27nessies = Qset1mask[QOFF+27]
			//					^ (Qprevmask [QOFF + 27] & q26)
			^ (Qprevrmask [QOFF + 27] & rotate_left(q26,30))
			//					^ (Qprev2rmask [QOFF + 27] & rotate_left(q25, 30))
			;
		bool valid_sol = 0 == ((q27 ^ q27nessies) & Qcondmask[QOFF + 27]);

		const uint32_t q28nessies = // Qset1mask[QOFF+28]
			//					^ (Qprevmask [QOFF + 28] & q27)
			//					^ (Qprevrmask [QOFF + 28] & rotate_left(q27, 30))
			/*^*/ (Qprev2rmask [QOFF + 28] & rotate_left(q26,30))
			;
		valid_sol &= 0 == ((q28 ^ q28nessies) & Qcondmask[QOFF + 28]);

		valid_sol &= 0 == ((q29 ^ Qset1mask[QOFF+29]) & Qcondmask[QOFF + 29]);
		valid_sol &= 0 == ((q30 ^ Qset1mask[QOFF+30]) & Qcondmask[QOFF + 30]);
		uint32_t sol2 = (q28_sol2 ^ (q12_bo<<16));
		Q30SOLBUF.write(Q30SOLCTL, valid_sol, q28_sol0, q28_sol1, sol2);

		q25 -= m24;
		m11 ^= q12_bo;
		m12 ^= q12_bo<<5;
		m16 ^= q12_bo>>2;
	}
	PERF_STOP_COUNTER(29);
}



__device__ void stepQ3140(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(31);
	using namespace dev;

	const uint32_t q30_sol0 = Q30SOLBUF.get<0>(thread_rd_idx);
	const uint32_t q30_sol1 = Q30SOLBUF.get<1>(thread_rd_idx);
	const uint32_t q30_sol2 = Q30SOLBUF.get<2>(thread_rd_idx);

	const uint32_t extsolidx = unpack_idx(q30_sol0,q30_sol1);
	const uint32_t basesolidx = Q20SOLBUF.get<10>(extsolidx);

	uint32_t m17 = Q20SOLBUF.get<6>(extsolidx);
	uint32_t m18 = Q20SOLBUF.get<7>(extsolidx);
	uint32_t m19 = Q20SOLBUF.get<8>(extsolidx);
	uint32_t m20 = Q20SOLBUF.get<9>(extsolidx);

	m17 ^= unpack_w17ext_nbs(q30_sol0,q30_sol1);
	m18 ^= unpack_w18_nbs(q30_sol0,q30_sol1);
	m19 &= ~W19NBPACKM;
	m19 ^= unpack_w19_nbs_fb(q30_sol0,q30_sol1);
	m20 &= ~W20NBPACKM;
	m20 ^= unpack_w20_nbs_fb(q30_sol0,q30_sol1);

	const uint32_t m5 = BASESOLBUF.get<6>(basesolidx);
	const uint32_t m7 = BASESOLBUF.get<8>(basesolidx);
	const uint32_t m13 = BASESOLBUF.get<14>(basesolidx);
	const uint32_t m21 = rotate_left(m18 ^ m13 ^ m7 ^ m5,1);

	uint32_t q19, q20, q21, q22, q23;
	{
		uint32_t q13 = BASESOLBUF.get<1>(basesolidx);
		uint32_t q14 = BASESOLBUF.get<2>(basesolidx);
		uint32_t q15 = Q20SOLBUF.get<0>(extsolidx);
		uint32_t q16 = Q20SOLBUF.get<1>(extsolidx);
		uint32_t q17 = Q20SOLBUF.get<2>(extsolidx);

		uint32_t q18  = sha1_round1(q17, q16, q15, q14, q13, m17); 
		q19  = sha1_round1(q18, q17, q16, q15, q14, m18); 
		q20  = sha1_round1(q19, q18, q17, q16, q15, m19); 
		q21  = sha1_round2(q20, q19, q18, q17, q16, m20); 
		q22  = sha1_round2(q21, q20, q19, q18, q17, m21); 
		q23  = sha1_round2(q22, q21, q20, q19, q18, 0); 
	}

	// booms change: m10, m11, m12, m14, m15, m16 => m22 (m14), m23 (m15), m24 (m10,m16), m25(m11), m26(m10,m12),m27(m11),m28(m12,m14),m29(m15)

	// m5
	const uint32_t m6 = BASESOLBUF.get<7>(basesolidx);
	// m7
	const uint32_t m8 = BASESOLBUF.get<9>(basesolidx);
	const uint32_t m9 = BASESOLBUF.get<10>(basesolidx);
	uint32_t m10 = BASESOLBUF.get<11>(basesolidx);
	uint32_t m11 = BASESOLBUF.get<12>(basesolidx);
	uint32_t m12 = BASESOLBUF.get<13>(basesolidx);
	// m13

	uint32_t m14 = Q20SOLBUF.get<3>(extsolidx);
	uint32_t m15 = Q20SOLBUF.get<4>(extsolidx);
	uint32_t m16 = Q20SOLBUF.get<5>(extsolidx);

	{
		const uint32_t q11boom = q30_sol2 & Q11BOOMS;
		const uint32_t q12boom = (q30_sol2>>16) & Q12BOOMS;
		m10 ^= q11boom;
		m11 ^= q11boom<<5;
		m14 ^= (q11boom>>2)&(1<<6);
		m15 ^= q11boom>>2;

		m11 ^= q12boom;
		m12 ^= q12boom<<5;
		m16 ^= q12boom>>2;
	}

	uint32_t m22 = rotate_left(m19 ^ m14 ^ m8 ^ m6,1);
	q23 += m22;

	const uint32_t m23 = rotate_left(m20 ^ m15 ^ m9 ^ m7, 1);
	const uint32_t m24 = rotate_left(m21 ^ m16 ^ m10 ^ m8, 1);
	const uint32_t m25 = rotate_left(m22 ^ m17 ^ m11 ^ m9, 1);
	const uint32_t m26 = rotate_left(m23 ^ m18 ^ m12 ^ m10, 1);
	const uint32_t m27 = rotate_left(m24 ^ m19 ^ m13 ^ m11, 1);
	const uint32_t m28 = rotate_left(m25 ^ m20 ^ m14 ^ m12, 1);
	const uint32_t m29 = rotate_left(m26 ^ m21 ^ m15 ^ m13, 1);

	const uint32_t q24 = rotate_left(q23,5) + sha1_f2(q22,rotate_left(q21,30),rotate_left(q20,30)) + rotate_left(q19,30) + 0x6ED9EBA1 + m23;
	const uint32_t q25 = rotate_left(q24,5) + sha1_f2(q23,rotate_left(q22,30),rotate_left(q21,30)) + rotate_left(q20,30) + 0x6ED9EBA1 + m24;
	uint32_t q26 = rotate_left(q25,5) + sha1_f2(q24,rotate_left(q23,30),rotate_left(q22,30)) + rotate_left(q21,30) + 0x6ED9EBA1 + m25;
	uint32_t q27 = rotate_left(q26,5) + sha1_f2(q25,rotate_left(q24,30),rotate_left(q23,30)) + rotate_left(q22,30) + 0x6ED9EBA1 + m26;
	uint32_t q28 = rotate_left(q27,5) + sha1_f2(q26,rotate_left(q25,30),rotate_left(q24,30)) + rotate_left(q23,30) + 0x6ED9EBA1 + m27;
	uint32_t q29 = rotate_left(q28,5) + sha1_f2(q27,rotate_left(q26,30),rotate_left(q25,30)) + rotate_left(q24,30) + 0x6ED9EBA1 + m28;
	uint32_t q30 = rotate_left(q29,5) + sha1_f2(q28,rotate_left(q27,30),rotate_left(q26,30)) + rotate_left(q25,30) + 0x6ED9EBA1 + m29;

	const uint32_t m30 = rotate_left(m27 ^ m22 ^ m16 ^ m14, 1);
	const uint32_t m31 = rotate_left(m28 ^ m23 ^ m17 ^ m15, 1);
	const uint32_t m32 = rotate_left(m29 ^ m24 ^ m18 ^ m16, 1);
	const uint32_t m33 = rotate_left(m30 ^ m25 ^ m19 ^ m17, 1);
	const uint32_t m34 = rotate_left(m31 ^ m26 ^ m20 ^ m18, 1);
	const uint32_t m35 = rotate_left(m32 ^ m27 ^ m21 ^ m19, 1);
	const uint32_t m36 = rotate_left(m33 ^ m28 ^ m22 ^ m20, 1);
	const uint32_t m37 = rotate_left(m34 ^ m29 ^ m23 ^ m21, 1);
	const uint32_t m38 = rotate_left(m35 ^ m30 ^ m24 ^ m22, 1);
	const uint32_t m39 = rotate_left(m36 ^ m31 ^ m25 ^ m23, 1);

	uint32_t q36,q37,q38,q39,q40;
	{
		const uint32_t q31 = rotate_left(q30,5) + sha1_f2(q29,rotate_left(q28,30),rotate_left(q27,30)) + rotate_left(q26,30) + 0x6ED9EBA1 + m30;
		const uint32_t q32 = rotate_left(q31,5) + sha1_f2(q30,rotate_left(q29,30),rotate_left(q28,30)) + rotate_left(q27,30) + 0x6ED9EBA1 + m31;
		const uint32_t q33 = rotate_left(q32,5) + sha1_f2(q31,rotate_left(q30,30),rotate_left(q29,30)) + rotate_left(q28,30) + 0x6ED9EBA1 + m32;
		const uint32_t q34 = rotate_left(q33,5) + sha1_f2(q32,rotate_left(q31,30),rotate_left(q30,30)) + rotate_left(q29,30) + 0x6ED9EBA1 + m33;
		const uint32_t q35 = rotate_left(q34,5) + sha1_f2(q33,rotate_left(q32,30),rotate_left(q31,30)) + rotate_left(q30,30) + 0x6ED9EBA1 + m34;
		q36 = rotate_left(q35,5) + sha1_f2(q34,rotate_left(q33,30),rotate_left(q32,30)) + rotate_left(q31,30) + 0x6ED9EBA1 + m35;
		q37 = rotate_left(q36,5) + sha1_f2(q35,rotate_left(q34,30),rotate_left(q33,30)) + rotate_left(q32,30) + 0x6ED9EBA1 + m36;
		q38 = rotate_left(q37,5) + sha1_f2(q36,rotate_left(q35,30),rotate_left(q34,30)) + rotate_left(q33,30) + 0x6ED9EBA1 + m37;
		q39 = rotate_left(q38,5) + sha1_f2(q37,rotate_left(q36,30),rotate_left(q35,30)) + rotate_left(q34,30) + 0x6ED9EBA1 + m38;
		q40 = rotate_left(q39,5) + sha1_f2(q38,rotate_left(q37,30),rotate_left(q36,30)) + rotate_left(q35,30) + 0x6ED9EBA1 + m39;
	}

	q26 += dQ[QOFF+26];
	q27 += dQ[QOFF+27];
	q28 += dQ[QOFF+28];
	q29 += dQ[QOFF+29];
	q30 += dQ[QOFF+30];
	uint32_t q36b,q37b,q38b,q39b,q40b;
	{
		const uint32_t q31 = rotate_left(q30,5) + sha1_f2(q29,rotate_left(q28,30),rotate_left(q27,30)) + rotate_left(q26,30) + 0x6ED9EBA1 + (m30^DV_DW[30]);
		const uint32_t q32 = rotate_left(q31,5) + sha1_f2(q30,rotate_left(q29,30),rotate_left(q28,30)) + rotate_left(q27,30) + 0x6ED9EBA1 + (m31^DV_DW[31]);
		const uint32_t q33 = rotate_left(q32,5) + sha1_f2(q31,rotate_left(q30,30),rotate_left(q29,30)) + rotate_left(q28,30) + 0x6ED9EBA1 + (m32^DV_DW[32]);
		const uint32_t q34 = rotate_left(q33,5) + sha1_f2(q32,rotate_left(q31,30),rotate_left(q30,30)) + rotate_left(q29,30) + 0x6ED9EBA1 + (m33^DV_DW[33]);
		const uint32_t q35 = rotate_left(q34,5) + sha1_f2(q33,rotate_left(q32,30),rotate_left(q31,30)) + rotate_left(q30,30) + 0x6ED9EBA1 + (m34^DV_DW[34]);
		q36b = rotate_left(q35,5) + sha1_f2(q34,rotate_left(q33,30),rotate_left(q32,30)) + rotate_left(q31,30) + 0x6ED9EBA1 + (m35^DV_DW[35]);
		q37b = rotate_left(q36b,5) + sha1_f2(q35,rotate_left(q34,30),rotate_left(q33,30)) + rotate_left(q32,30) + 0x6ED9EBA1 + (m36^DV_DW[36]);
		q38b = rotate_left(q37b,5) + sha1_f2(q36b,rotate_left(q35,30),rotate_left(q34,30)) + rotate_left(q33,30) + 0x6ED9EBA1 + (m37^DV_DW[37]);
		q39b = rotate_left(q38b,5) + sha1_f2(q37b,rotate_left(q36b,30),rotate_left(q35,30)) + rotate_left(q34,30) + 0x6ED9EBA1 + (m38^DV_DW[38]);
		q40b = rotate_left(q39b,5) + sha1_f2(q38b,rotate_left(q37b,30),rotate_left(q36b,30)) + rotate_left(q35,30) + 0x6ED9EBA1 + (m39^DV_DW[39]);
	}
	bool valid_sol = 0 == (  (q36^q36b) | (q37^q37b) | (q38^q38b) | (q39^q39b) | (q40^q40b) );

	Q40SOLBUF.write(Q40SOLCTL, valid_sol, q36, q37, q38, q39, q40, m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35,m36,m37,m38,m39);

	PERF_STOP_COUNTER(31);
}



__device__ void step_extend_Q60(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(60);
	using namespace dev;

	uint32_t e = Q40SOLBUF.get<0>(thread_rd_idx); // q36
	uint32_t d = Q40SOLBUF.get<1>(thread_rd_idx); // q37
	uint32_t c = Q40SOLBUF.get<2>(thread_rd_idx); // q38
	uint32_t b = Q40SOLBUF.get<3>(thread_rd_idx); // q39
	uint32_t a = Q40SOLBUF.get<4>(thread_rd_idx); // q40
	uint32_t E = e;
	uint32_t D = d;
	uint32_t C = c;
	uint32_t B = b;
	uint32_t A = a;

	uint32_t m40, m41, m42, m43, m44, m45, m46, m47;
	uint32_t m48, m49, m50, m51, m52, m53, m54, m55;
	uint32_t m56, m57, m58, m59;

	{
		uint32_t m24 = Q40SOLBUF.get<5>(thread_rd_idx);
		uint32_t m25 = Q40SOLBUF.get<6>(thread_rd_idx);
		uint32_t m26 = Q40SOLBUF.get<7>(thread_rd_idx);
		uint32_t m27 = Q40SOLBUF.get<8>(thread_rd_idx);
		uint32_t m28 = Q40SOLBUF.get<9>(thread_rd_idx);
		uint32_t m29 = Q40SOLBUF.get<10>(thread_rd_idx);
		uint32_t m30 = Q40SOLBUF.get<11>(thread_rd_idx);
		uint32_t m31 = Q40SOLBUF.get<12>(thread_rd_idx);
		uint32_t m32 = Q40SOLBUF.get<13>(thread_rd_idx);
		uint32_t m33 = Q40SOLBUF.get<14>(thread_rd_idx);
		uint32_t m34 = Q40SOLBUF.get<15>(thread_rd_idx);
		uint32_t m35 = Q40SOLBUF.get<16>(thread_rd_idx);
		uint32_t m36 = Q40SOLBUF.get<17>(thread_rd_idx);
		uint32_t m37 = Q40SOLBUF.get<18>(thread_rd_idx);
		uint32_t m38 = Q40SOLBUF.get<19>(thread_rd_idx);
		uint32_t m39 = Q40SOLBUF.get<20>(thread_rd_idx);

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
		m56 = sha1_mess(m53, m48, m42, m40);
		m57 = sha1_mess(m54, m49, m43, m41);
		m58 = sha1_mess(m55, m50, m44, m42);
		m59 = sha1_mess(m56, m51, m45, m43);
	}

	e = sha1_round3(a, b, c, d, e, m40);
	d = sha1_round3(e, a, b, c, d, m41);
	c = sha1_round3(d, e, a, b, c, m42);
	b = sha1_round3(c, d, e, a, b, m43);
	a = sha1_round3(b, c, d, e, a, m44);

	e = sha1_round3(a, b, c, d, e, m45);
	d = sha1_round3(e, a, b, c, d, m46);
	c = sha1_round3(d, e, a, b, c, m47);
	b = sha1_round3(c, d, e, a, b, m48);
	a = sha1_round3(b, c, d, e, a, m49);

	e = sha1_round3(a, b, c, d, e, m50);
	d = sha1_round3(e, a, b, c, d, m51);
	c = sha1_round3(d, e, a, b, c, m52);
	b = sha1_round3(c, d, e, a, b, m53);
	a = sha1_round3(b, c, d, e, a, m54);

	e = sha1_round3(a, b, c, d, e, m55);
	d = sha1_round3(e, a, b, c, d, m56);
	c = sha1_round3(d, e, a, b, c, m57);
	b = sha1_round3(c, d, e, a, b, m58);
	a = sha1_round3(b, c, d, e, a, m59);

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
	m56 ^= DV_DW[56];
	m57 ^= DV_DW[57];
	m58 ^= DV_DW[58];
	m59 ^= DV_DW[59];

	E = sha1_round3(A, B, C, D, E, m40);
	D = sha1_round3(E, A, B, C, D, m41);
	C = sha1_round3(D, E, A, B, C, m42);
	B = sha1_round3(C, D, E, A, B, m43);
	A = sha1_round3(B, C, D, E, A, m44);

	E = sha1_round3(A, B, C, D, E, m45);
	D = sha1_round3(E, A, B, C, D, m46);
	C = sha1_round3(D, E, A, B, C, m47);
	B = sha1_round3(C, D, E, A, B, m48);
	A = sha1_round3(B, C, D, E, A, m49);

	E = sha1_round3(A, B, C, D, E, m50);
	D = sha1_round3(E, A, B, C, D, m51);
	C = sha1_round3(D, E, A, B, C, m52);
	B = sha1_round3(C, D, E, A, B, m53);
	A = sha1_round3(B, C, D, E, A, m54);

	E = sha1_round3(A, B, C, D, E, m55);
	D = sha1_round3(E, A, B, C, D, m56);
	C = sha1_round3(D, E, A, B, C, m57);
	B = sha1_round3(C, D, E, A, B, m58);
	A = sha1_round3(B, C, D, E, A, m59);

//	bool good60 = (e == E);
//	good60 = (d == D) && good60;
//	good60 = (c == C) && good60;
//	good60 = (b == B) && good60;
//	good60 = (a == A) && good60;

	bool good60 = 0 == ((e^E)|(d^D)|(c^C)|(b^B)|(a^A));
	// sol: Q56,..,Q60,m44,...,m59
	COLLCANDIDATEBUF.write(COLLCANDIDATECTL, good60, e, d, c, b, a,
											 m44, m45, m46, m47, m48, m49, m50, m51,
											 m52, m53, m54, m55, m56, m57, m58, m59);
	PERF_STOP_COUNTER(60);
}









// BACKUP CONTROLS ONLY IF THEY ARE IN SHARED (AND THUS BLOCK-SPECIFIC)
__device__ void backup_controls()
{
	__syncthreads();
	if (threadIdx.x == 0)
	{
//		q18_solutions_ctl_bu[blockIdx.x] = Q18SOLCTL;
//		q19_solutions_ctl_bu[blockIdx.x] = Q19SOLCTL;
		q20_solutions_ctl_bu[blockIdx.x] = Q20SOLCTL;
		q21_solutions_ctl_bu[blockIdx.x] = Q21SOLCTL;
		q22_solutions_ctl_bu[blockIdx.x] = Q22SOLCTL;

		q23_solutions_ctl_bu[blockIdx.x] = Q23SOLCTL;
		q26_solutions_ctl_bu[blockIdx.x] = Q26SOLCTL;
		q28_solutions_ctl_bu[blockIdx.x] = Q28SOLCTL;
		q30_solutions_ctl_bu[blockIdx.x] = Q30SOLCTL;
		q40_solutions_ctl_bu[blockIdx.x] = Q40SOLCTL;

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
//		Q18SOLCTL = q18_solutions_ctl_bu[blockIdx.x];
//		Q19SOLCTL = q19_solutions_ctl_bu[blockIdx.x];
		Q20SOLCTL = q20_solutions_ctl_bu[blockIdx.x];
		Q21SOLCTL = q21_solutions_ctl_bu[blockIdx.x];
		Q22SOLCTL = q22_solutions_ctl_bu[blockIdx.x];

		Q23SOLCTL = q23_solutions_ctl_bu[blockIdx.x];
		Q26SOLCTL = q26_solutions_ctl_bu[blockIdx.x];
		Q28SOLCTL = q28_solutions_ctl_bu[blockIdx.x];
		Q30SOLCTL = q30_solutions_ctl_bu[blockIdx.x];
		Q40SOLCTL = q40_solutions_ctl_bu[blockIdx.x];

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
	Q21SOLBUF.reset(Q21SOLCTL);
	Q22SOLBUF.reset(Q22SOLCTL);

	Q23SOLBUF.reset(Q23SOLCTL);
	Q26SOLBUF.reset(Q26SOLCTL);
	Q28SOLBUF.reset(Q28SOLCTL);
	Q30SOLBUF.reset(Q30SOLCTL);
	Q40SOLBUF.reset(Q40SOLCTL);

	COLLCANDIDATEBUF.reset(COLLCANDIDATECTL);

#ifdef USE_PERF_COUNTERS
	performance_reset();
#endif
	backup_controls();
}

__global__ void cuda_attack()
{
	restore_controls();
	
	__shared__ uint32_t stepQ18done;
	__shared__ uint64_t startclock;
	if (threadIdx.x==0)
	{
		stepQ18done = 0;
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
			uint32_t thidx = Q40SOLBUF.getreadidx(Q40SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				step_extend_Q60(thidx);
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q30SOLBUF.getreadidx(Q30SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ3140(thidx);
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q28SOLBUF.getreadidx(Q28SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ2930(thidx);
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q26SOLBUF.getreadidx(Q26SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ2728(thidx);
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q23SOLBUF.getreadidx(Q23SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ2456(thidx);
				DEBUG_BREAK;
				continue;
			}
		}
#endif




#if 1
		{
			uint32_t thidx = Q22SOLBUF.getreadidx(Q22SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ23(thidx);
				DEBUG_BREAK;
				continue;
			}
		}
#endif





#if 1
		{
			uint32_t thidx = Q21SOLBUF.getreadidx(Q21SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ22(thidx);
				DEBUG_BREAK;
				continue;
			}
		}
#endif


#if 1
		{
			uint32_t thidx = Q20SOLBUF.getreadidx(Q20SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ21(thidx);
				DEBUG_BREAK;
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
				DEBUG_BREAK;
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
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		// only let warp 0 of each block grab a basesol once per kernel run
		if (stepQ18done == 0 && (threadIdx.x>>5)==0)
		{
			uint32_t thidx = BASESOLBUF.getreadidx(BASESOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				if (threadIdx.x == 0)
					stepQ18done = 1;
				stepQ18(thidx);
		//		break;
				DEBUG_BREAK;
				continue;
			}
		}
#endif

	}
	}
#ifdef USE_CLOCK_LOOP
#ifdef USE_PERF_COUNTERS
	while ((clock64()-startclock) < (uint64_t(1)<<32));
#else
	while ((clock64()-startclock) < (uint64_t(1)<<34));
#endif
#endif


	backup_controls();
}




void verify_step_computations(int cuda_blocks);
bool verify(basesol_t basesol);
void print_attack_info();


void save_q60solutions()
{
	if (outputfile.empty())
	{
		outputfile = "q60sols.txt";
//		return;
	}
	static size_t oldsize = 0;

 	q60sol_t sol;
	vector<q60sol_t> q60sols;
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
		q60sols.push_back(sol);
	}

	if (oldsize != q60sols.size())
	{
		oldsize = q60sols.size();
		cout << "Writing " << q60sols.size() << " Q60-solutions to '" << outputfile << "'..." << endl;

		ofstream ofs(outputfile.c_str());
		if (!ofs)
		{
			cout << "Error opening '" << outputfile << ".tmp'!" << endl;
			return;
		}
		for (size_t i = 0; i < q60sols.size(); ++i)
		{
			ofs << encode_q60sol(q60sols[i]) << endl;
		}
	}
}

buffer_basesol_t  basesol_buf_host;
control_basesol_t basesol_ctl_host;







void cuda_main(std::vector<basesol_t>& basesols)
{
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
	if (prop.regsPerBlock/64 < cuda_threads_per_block)
	{
		cuda_threads_per_block = prop.regsPerBlock/64;
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

#if 1
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
		// Q12,..,Q17,m4,...,m19
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


	// use auto for same type deduction, same type is necessary for proper wrap-around behaviour
	uint32_t q18oldbufsize;
	uint32_t q19oldbufsize;
	uint32_t q20oldbufsize[BLOCKS];
	uint32_t q21oldbufsize[BLOCKS];
	uint32_t q22oldbufsize[BLOCKS];
	uint32_t q23oldbufsize[BLOCKS];
	uint32_t q26oldbufsize[BLOCKS];
	uint32_t q28oldbufsize[BLOCKS];
	uint32_t q30oldbufsize[BLOCKS];
	uint32_t q40oldbufsize[BLOCKS];
	q18oldbufsize = q18_solutions_ctl.write_idx;
	q19oldbufsize = q19_solutions_ctl.write_idx;
	for (unsigned bl = 0; bl < cuda_blocks; ++bl)
	{
		q20oldbufsize[bl] = q20_solutions_ctl_bu[bl].write_idx;
		q21oldbufsize[bl] = q21_solutions_ctl_bu[bl].write_idx;
		q22oldbufsize[bl] = q22_solutions_ctl_bu[bl].write_idx;
		q23oldbufsize[bl] = q23_solutions_ctl_bu[bl].write_idx;
		q26oldbufsize[bl] = q26_solutions_ctl_bu[bl].write_idx;
		q28oldbufsize[bl] = q28_solutions_ctl_bu[bl].write_idx;
		q30oldbufsize[bl] = q30_solutions_ctl_bu[bl].write_idx;
		q40oldbufsize[bl] = q40_solutions_ctl_bu[bl].write_idx;
	}
	uint64_t q18sols = 0, q19sols = 0, q20sols = 0, q21sols = 0, q22sols = 0, q23sols = 0, q26sols = 0, q28sols = 0, q30sols = 0, q40sols = 0, q60sols = 0;


	cout << "Starting CUDA kernel" << flush;
	hc::timer cuda_total_time(true);
	while (true)
	{
		cout << "." << flush;

		hc::timer cuda_time(true);
		cuda_attack<<<cuda_blocks,cuda_threads_per_block>>>();
		CUDA_ASSERT( cudaDeviceSynchronize() );
		cout << "CUDA running time: " << cuda_time.time() << endl;

#ifdef VERIFY_GPU_RESULTS
		verify_step_computations(cuda_blocks);
#endif

		uint64_t basesolsleft = uint32_t(base_solutions_ctl.write_idx - base_solutions_ctl.read_idx);
		uint64_t gl_workleft_base = uint32_t(basesolsleft)>>5;
		uint64_t gl_workleft_q18 = ((q18_solutions_ctl.write_idx - q18_solutions_ctl.read_idx) % q18_solutions_ctl.size) >>5;
		uint64_t gl_workleft_q19 = ((q19_solutions_ctl.write_idx - q19_solutions_ctl.read_idx) % q19_solutions_ctl.size) >>5;
		uint64_t gl_workleft = gl_workleft_base + gl_workleft_q18 + gl_workleft_q19;
		
		q18sols += q18_solutions_ctl.write_idx - q18oldbufsize;
		q19sols += q19_solutions_ctl.write_idx - q19oldbufsize;
		q18oldbufsize = q18_solutions_ctl.write_idx;
		q19oldbufsize = q19_solutions_ctl.write_idx;
		
		uint64_t workleft = gl_workleft;
		for (unsigned bl = 0; bl < cuda_blocks; ++bl)
		{
			workleft += (q20_solutions_ctl_bu[bl].write_idx - q20_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q21_solutions_ctl_bu[bl].write_idx - q21_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q22_solutions_ctl_bu[bl].write_idx - q22_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q23_solutions_ctl_bu[bl].write_idx - q23_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q26_solutions_ctl_bu[bl].write_idx - q26_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q28_solutions_ctl_bu[bl].write_idx - q28_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q30_solutions_ctl_bu[bl].write_idx - q30_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q40_solutions_ctl_bu[bl].write_idx - q40_solutions_ctl_bu[bl].read_idx)>>5;

			q20sols += q20_solutions_ctl_bu[bl].write_idx - q20oldbufsize[bl];
			q21sols += q21_solutions_ctl_bu[bl].write_idx - q21oldbufsize[bl];
			q22sols += q22_solutions_ctl_bu[bl].write_idx - q22oldbufsize[bl];
			q23sols += q23_solutions_ctl_bu[bl].write_idx - q23oldbufsize[bl];
			q26sols += q26_solutions_ctl_bu[bl].write_idx - q26oldbufsize[bl];
			q28sols += q28_solutions_ctl_bu[bl].write_idx - q28oldbufsize[bl];
			q30sols += q30_solutions_ctl_bu[bl].write_idx - q30oldbufsize[bl];
			q40sols += q40_solutions_ctl_bu[bl].write_idx - q40oldbufsize[bl];
			q20oldbufsize[bl] = q20_solutions_ctl_bu[bl].write_idx;
			q21oldbufsize[bl] = q21_solutions_ctl_bu[bl].write_idx;
			q22oldbufsize[bl] = q22_solutions_ctl_bu[bl].write_idx;
			q23oldbufsize[bl] = q23_solutions_ctl_bu[bl].write_idx;
			q26oldbufsize[bl] = q26_solutions_ctl_bu[bl].write_idx;
			q28oldbufsize[bl] = q28_solutions_ctl_bu[bl].write_idx;
			q30oldbufsize[bl] = q30_solutions_ctl_bu[bl].write_idx;
			q40oldbufsize[bl] = q40_solutions_ctl_bu[bl].write_idx;
		}
		q60sols = collision_candidates_ctl.write_idx;
		cout << "Q18 sols:\t" << q18sols << "\t" << (double(q18sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q19 sols:\t" << q19sols << "\t" << (double(q19sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q20 sols:\t" << q20sols << "\t" << (double(q20sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q21 sols:\t" << q21sols << "\t" << (double(q21sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q22 sols:\t" << q22sols << "\t" << (double(q22sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q23 sols:\t" << q23sols << "\t" << (double(q23sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q26 sols:\t" << q26sols << "\t" << (double(q26sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q28 sols:\t" << q28sols << "\t" << (double(q28sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q30 sols:\t" << q30sols << "\t" << (double(q30sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q40 sols:\t" << q40sols << "\t" << (double(q40sols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q60 sols:\t" << q60sols << "\t" << (double(q60sols)/cuda_total_time.time()) << "#/s" << endl;

		save_q60solutions();

#ifdef USE_PERF_COUNTERS
		show_performance_counters();
#endif

		// exit if base solutions have been exhausted
		// !! NOTE THAT THERE MAY STILL BE SOME OTHER WORK LEFT !!
		
		cout << "Basesolutions left: " << basesolsleft << "\t" << (double(base_solutions_ctl.read_idx)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Global work left: B:" << gl_workleft_base << "  Q18:" << gl_workleft_q18 << "  Q19:" << gl_workleft_q19 << endl;
		
		if (gl_workleft == 0)
		{
			cout << "Exhausted work!" << endl;
			break;
		}

//		boost::this_thread::sleep_for( boost::chrono::seconds(1) );
	}
}
































int cores_per_mp(int cc)
{
	switch (cc)
	{
		case 0x10: // TESLA G80
		case 0x11: // TESLA G8x
		case 0x12: // TESLA G9x
		case 0x13: // TESLA GT200
			return 8;
		case 0x20: // FERMI GF100
			return 32;
		case 0x21: // FERMI GF10x
			return 48;
		case 0x30: // KEPLER GK10x
		case 0x32: // KEPLER GK10x
		case 0x35: // KEPLER GK11x
		case 0x37: // KEPLER GK21x
			return 192;
		case 0x50: // MAXWELL GM10x
		case 0x52: // MAXWELL GM20x
			return 128;
		default: // unknown
			return -1;
	}
}
void cuda_query()
{
	cout << "======== CUDA DEVICE QUERY ======== " << endl;
	int devicecount = 0;
	CUDA_ASSERT( cudaGetDeviceCount(&devicecount) );
	cout << "Detected " << devicecount << " CUDA Capable device(s)." << endl;
	for (int dev = 0; dev < devicecount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, dev);
		cout << "======== device " << dev << " : " << prop.name << " ========" << endl;
		int driverversion, runtimeversion;
		cudaDriverGetVersion(&driverversion);
		cudaRuntimeGetVersion(&runtimeversion);
		cout << "CUDA Driver       : " << (driverversion/1000) << "." << (driverversion%100)/10 << endl;
		cout << "CUDA Runtime      : " << (runtimeversion/1000) << "." << (runtimeversion%100)/10 << endl;
		cout << "CUDA Capability   : " << prop.major << "." << prop.minor << endl;
		cout << "Global memory     : " << prop.totalGlobalMem << " bytes" << endl;
		cout << "Cores             : " << prop.multiProcessorCount << " MP x " << cores_per_mp((prop.major<<4)+prop.minor) << " cores/MP = " << prop.multiProcessorCount*cores_per_mp((prop.major<<4)+prop.minor) << " cores" << endl;
		cout << "Clock rate        : " << prop.clockRate * 1e-3f << " MHz" << endl;
		cout << "Constant mem      : " << prop.totalConstMem << " bytes" << endl;
		cout << "Shared mem/Block  : " << prop.sharedMemPerBlock << " bytes" << endl;
		cout << "Registers /Block  : " << prop.regsPerBlock << endl;
		cout << "MaxThreads/MP     : " << prop.maxThreadsPerMultiProcessor << endl;
		cout << "MaxThreads/Block  : " << prop.maxThreadsPerBlock << endl;
		cout << "Runtime limit     : " << (prop.kernelExecTimeoutEnabled?"YES":"NO") << endl;
		cout << "Unified Addressing: " << (prop.unifiedAddressing?"YES":"NO") << endl;
		cout << endl;
	}
}









#ifdef VERIFY_GPU_RESULTS



#define VERIFY_ERROR(s) { cout << "Err @ block=" << block << " bufidx=" << read_idx << " baseidx=" << base_idx << " : " << s << endl; ok = false; }

uint32_t isQokay(int t, uint32_t Q[])
{
	using namespace host;
	uint32_t Qval = Qset1mask[QOFF+t]
		^ (Qnextmask[QOFF+t] & Q[QOFF+t+1])
		^ (Qprevmask[QOFF+t] & Q[QOFF+t-1])
		^ (Qprevrmask[QOFF+t] & rotate_left(Q[QOFF+t-1],30))
		^ (Qprev2rmask[QOFF+t] & rotate_left(Q[QOFF+t-2],30))
		;
	return ( (Q[QOFF+t] ^ Qval) & Qcondmask[QOFF+t] );
}

vector<uint32_t>& operator^=(vector<uint32_t>& l, const vector<uint32_t>& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("vector<uint32_t>::operator^=(): unequal sizes!");
	for (unsigned i = 0; i < l.size(); ++i)
		l[i] ^= r[i];
	return l;
}

void print_convert_msgbitrel(const uint32_t rel[17], int oldmainblockoffset, int newmainblockoffset)
{
	vector<uint32_t> bitrel(81, 0);
	for (int t = oldmainblockoffset; t < oldmainblockoffset + 16; ++t)
		bitrel[t] = rel[t - oldmainblockoffset];
	bitrel[80] = rel[16];

	map< vector<uint32_t>, vector<uint32_t> > bit_exp;
	for (unsigned t = 0; t < 16; ++t)
		for (unsigned b = 0; b < 32; ++b)
		{
			vector<uint32_t> l(16, 0), r(80, 0);
			l[t] |= 1 << b;
			sha1_me_generalised(&r[0], &l[0], newmainblockoffset);
			bit_exp[l] = r;
		}

	map< pair<unsigned, unsigned>, vector<uint32_t> > bit80_vec16;
	for (unsigned t = 0; t < 80; ++t)
		for (unsigned b = 0; b < 32; ++b)
		{
			vector<uint32_t> r(16, 0);
			for (map<vector<uint32_t>,vector<uint32_t> >::const_iterator it = bit_exp.begin(); it != bit_exp.end(); ++it)
				if (it->second[t] & (1 << b))
					r ^= it->first;
			bit80_vec16[make_pair(t, b)] = r;
		}

	vector<uint32_t> newrel(16);
	for (unsigned t = 0; t < 80; ++t)
		for (unsigned b = 0; b < 32; ++b)
			if (bitrel[t] & (1 << b))
				newrel ^= bit80_vec16[make_pair(t, b)];
	newrel.resize(17);
	newrel[16] = (bitrel[80] == 0) ? 0 : 1;

	for (unsigned t = newmainblockoffset; t < newmainblockoffset + 16; ++t)
		for (unsigned b = 0; b < 32; ++b)
			if (newrel[t - newmainblockoffset] & (1 << b))
				cout << " ^ W" << t << "[" << b << "]";
	cout << " = " << (newrel[16] & 1) << endl;
}







bool verify_Q18_Q19(int block, size_t read_idx, int lastQ, uint32_t w0, uint32_t w1)
{
	bool ok = true;
	using namespace host;

	uint32_t m[80];
	uint32_t Q[85];

	size_t base_idx = unpack_idx(w0,w1);
	Q[QOFF+12] = base_solutions_buf.get<0>(base_idx);
	Q[QOFF+13] = base_solutions_buf.get<1>(base_idx);
	Q[QOFF+14] = base_solutions_buf.get<2>(base_idx);
	Q[QOFF+15] = base_solutions_buf.get<3>(base_idx);
	Q[QOFF+16] = base_solutions_buf.get<4>(base_idx);
	Q[QOFF+17] = base_solutions_buf.get<5>(base_idx);
	m[ 5] = base_solutions_buf.get< 6>(base_idx);
	m[ 6] = base_solutions_buf.get< 7>(base_idx);
	m[ 7] = base_solutions_buf.get< 8>(base_idx);
	m[ 8] = base_solutions_buf.get< 9>(base_idx);
	m[ 9] = base_solutions_buf.get<10>(base_idx);
	m[10] = base_solutions_buf.get<11>(base_idx);
	m[11] = base_solutions_buf.get<12>(base_idx);
	m[12] = base_solutions_buf.get<13>(base_idx);
	m[13] = base_solutions_buf.get<14>(base_idx);
	m[14] = base_solutions_buf.get<15>(base_idx);
	m[15] = base_solutions_buf.get<16>(base_idx);
	m[16] = base_solutions_buf.get<17>(base_idx);
	m[17] = base_solutions_buf.get<18>(base_idx);
	m[18] = base_solutions_buf.get<19>(base_idx);
	m[19] = base_solutions_buf.get<20>(base_idx);
	m[20] = base_solutions_buf.get<21>(base_idx);

	uint32_t m14nb = unpack_w14_nbs(w0, w1);
	uint32_t m15nb = unpack_w15_nbs(w0, w1);
	uint32_t m16nb = unpack_w16_nbs(w0, w1);
	uint32_t m17nb = unpack_w17base_nbs(w0, w1);
	if (m14nb & ~W14NBALLM) VERIFY_ERROR("m14nb bad");
	if (m15nb & ~W15NBALLM) VERIFY_ERROR("m15nb bad");
	if (m16nb & ~W16NBALLM) VERIFY_ERROR("m16nb bad");
	if (m17nb & ~W17NBBASM) VERIFY_ERROR("m17nb bad");

	m[14] |= m14nb;
	m[15] |= m15nb;
	m[16] |= m16nb;
	m[17] |= m17nb;


	uint32_t* main_m1 = m;
	//w18[5]:  W15[4]
	main_m1[18] ^= ((0 ^ (main_m1[15] >> 4)) & 1) << 5;

	//w19[13]:  W15[15]  W15[16]  W17[14]  W17[16]  W17[19]
	main_m1[19] ^= ((0 ^ (main_m1[15] >> 15) ^ (main_m1[15] >> 16) ^ (main_m1[17] >> 14) ^ (main_m1[17] >> 16) ^ (main_m1[17] >> 19)) & 1) << 13;

	//w19[15] : W14[10]  W15[12]  W15[13]  W16[13]  W16[14]  W16[15]  W17[11]  W17[13]  W17[16]  W17[18]  W18[12]  W19[10]
	main_m1[19] ^= ((0 ^ (main_m1[14] >> 10) ^ (main_m1[15] >> 12) ^ (main_m1[15] >> 13) ^ (main_m1[16] >> 13) ^ (main_m1[16] >> 14) ^ (main_m1[16] >> 15) ^ (main_m1[17] >> 11) ^ (main_m1[17] >> 13) ^ (main_m1[17] >> 16) ^ (main_m1[17] >> 18) ^ (main_m1[18] >> 12) ^ (main_m1[19] >> 10)) & 1) << 15;

	//w19[16] : W17[17]  W17[19]
	main_m1[19] ^= ((0 ^ (main_m1[17] >> 17) ^ (main_m1[17] >> 19)) & 1) << 16;

	//w19[17] : W16[16]  W17[18]  W17[19]  W18[15]
	main_m1[19] ^= ((0 ^ (main_m1[16] >> 16) ^ (main_m1[17] >> 18) ^ (main_m1[17] >> 19) ^ (main_m1[18] >> 15)) & 1) << 17;

	//w19[18] : W17[19]
	main_m1[19] ^= ((0 ^ (main_m1[17] >> 19)) & 1) << 18;

	//w20[0] : W16[16]  W17[19]  W18[15]
	main_m1[20] ^= ((0 ^ (main_m1[16] >> 16) ^ (main_m1[17] >> 19) ^ (main_m1[18] >> 15)) & 1) << 0;

	//w20[14] : W15[14]  W15[16]  W17[18]  W17[19]
	main_m1[20] ^= ((0 ^ (main_m1[15] >> 14) ^ (main_m1[15] >> 16) ^ (main_m1[17] >> 18) ^ (main_m1[17] >> 19)) & 1) << 14;

	//w20[16] : W15[16]
	main_m1[20] ^= ((0 ^ (main_m1[15] >> 16)) & 1) << 16;

	//w20[17] : W15[16]  W16[13]  W16[15]  W17[16]  W18[12]
	main_m1[20] ^= ((0 ^ (main_m1[15] >> 16) ^ (main_m1[16] >> 13) ^ (main_m1[16] >> 15) ^ (main_m1[17] >> 16) ^ (main_m1[18] >> 12)) & 1) << 17;

	Q[QOFF + 16] -= rotate_left(Q[QOFF + 15], 5);

	Q[QOFF + 15] += m14nb;

	Q[QOFF + 16] += rotate_left(Q[QOFF + 15], 5);
	Q[QOFF + 16] += m15nb;

	for (int t = 16; t < lastQ; ++t)
	{
		sha1_step(t, Q, m);
	}

	for (int t = 14; t <= lastQ; ++t)
	{
		uint32_t okm = isQokay(t,Q);
		if (okm != 0)
		{
			VERIFY_ERROR("Q" << t << " bad !" << hex << okm << dec);
			ok = false;
		}
	}
	return ok;
}





bool verify_Q20(int block, size_t read_idx)
{
	bool ok = true;
	using namespace host;
	const int lastQ = 20;

	uint32_t m[80];
	uint32_t Q[85];

	// extsol: q15, q16, q17, m14, m15, m16, m17, m18, m19, m20, base_idx
	size_t base_idx = q20_solutions_buf[block].get<10>(read_idx);
	
	Q[QOFF + 12] = base_solutions_buf.get<0>(base_idx);
	Q[QOFF + 13] = base_solutions_buf.get<1>(base_idx);
	Q[QOFF + 14] = base_solutions_buf.get<2>(base_idx);

	Q[QOFF + 15] = q20_solutions_buf[block].get<0>(read_idx);
	Q[QOFF + 16] = q20_solutions_buf[block].get<1>(read_idx);
	Q[QOFF + 17] = q20_solutions_buf[block].get<2>(read_idx);

	m[ 5] = base_solutions_buf.get< 6>(base_idx);
	m[ 6] = base_solutions_buf.get< 7>(base_idx);
	m[ 7] = base_solutions_buf.get< 8>(base_idx);
	m[ 8] = base_solutions_buf.get< 9>(base_idx);
	m[ 9] = base_solutions_buf.get<10>(base_idx);
	m[10] = base_solutions_buf.get<11>(base_idx);
	m[11] = base_solutions_buf.get<12>(base_idx);
	m[12] = base_solutions_buf.get<13>(base_idx);
	m[13] = base_solutions_buf.get<14>(base_idx);

	m[14]        = q20_solutions_buf[block].get<3>(read_idx);
	m[15]        = q20_solutions_buf[block].get<4>(read_idx);
	m[16]        = q20_solutions_buf[block].get<5>(read_idx);
	m[17]        = q20_solutions_buf[block].get<6>(read_idx);
	m[18]        = q20_solutions_buf[block].get<7>(read_idx);
	m[19]        = q20_solutions_buf[block].get<8>(read_idx);
	m[20]        = q20_solutions_buf[block].get<9>(read_idx);

	sha1_me_generalised(m, 5);

	// compute previous steps
	for (int t = 16; t >= 0; --t)
		sha1_step_bw(t, Q, m);
	for (int t = 17; t < lastQ; ++t)
		sha1_step(t, Q, m);

	// verify stateconditions Q-2,...,Q20
	for (int t = -4+2; t <= lastQ; ++t)
	{
		uint32_t okm = isQokay(t, Q);
		if (okm != 0)
		{
			VERIFY_ERROR("Q" << t << " bad !" << hex << okm << dec);
			ok = false;
		}
	}

	// verify msgbitrelations
	// [1200] verify message bitrelations
	for (unsigned r = 0; r < msgbitrels16_size; ++r)
	{
		uint32_t w = msgbitrels16[r][16];
		for (unsigned t = mainblockoffset; t < mainblockoffset + 16; ++t)
		{
			w ^= m[t] & msgbitrels16[r][t - mainblockoffset];
		}
		if (0 != (hc::hw(w) & 1))
		{
			std::cerr << "bitrelation " << r << " is not satisfied!" << std::endl;
			print_convert_msgbitrel(msgbitrels16[r], 1, 5);
			return false;
		}
	}

	return ok;
}












bool verify_Q21_Q26(int block, size_t read_idx, int lastQ, uint32_t w0, uint32_t w1)
{
	bool ok = true;
	using namespace host;

	uint32_t m[80];
	uint32_t Q[85];

	size_t ext_idx = unpack_idx(w0,w1);
	size_t base_idx = q20_solutions_buf[block].get<10>(ext_idx);

	Q[QOFF + 12] = base_solutions_buf.get<0>(base_idx);
	Q[QOFF + 13] = base_solutions_buf.get<1>(base_idx);
	Q[QOFF + 14] = base_solutions_buf.get<2>(base_idx);

	Q[QOFF + 15] = q20_solutions_buf[block].get<0>(ext_idx);
	Q[QOFF + 16] = q20_solutions_buf[block].get<1>(ext_idx);
	Q[QOFF + 17] = q20_solutions_buf[block].get<2>(ext_idx);

	m[ 5] = base_solutions_buf.get< 6>(base_idx);
	m[ 6] = base_solutions_buf.get< 7>(base_idx);
	m[ 7] = base_solutions_buf.get< 8>(base_idx);
	m[ 8] = base_solutions_buf.get< 9>(base_idx);
	m[ 9] = base_solutions_buf.get<10>(base_idx);
	m[10] = base_solutions_buf.get<11>(base_idx);
	m[11] = base_solutions_buf.get<12>(base_idx);
	m[12] = base_solutions_buf.get<13>(base_idx);
	m[13] = base_solutions_buf.get<14>(base_idx);

	m[14]        = q20_solutions_buf[block].get<3>(ext_idx);
	m[15]        = q20_solutions_buf[block].get<4>(ext_idx);
	m[16]        = q20_solutions_buf[block].get<5>(ext_idx);
	m[17]        = q20_solutions_buf[block].get<6>(ext_idx);
	m[18]        = q20_solutions_buf[block].get<7>(ext_idx);
	m[19]        = q20_solutions_buf[block].get<8>(ext_idx);
	m[20]        = q20_solutions_buf[block].get<9>(ext_idx);

	m[17] ^= unpack_w17ext_nbs(w0,w1);
	m[18] ^= unpack_w18_nbs(w0,w1);
	m[19] &= ~W19NBPACKM;
	m[19] ^= unpack_w19_nbs_fb(w0,w1);
	m[20] &= ~W20NBPACKM;
	m[20] ^= unpack_w20_nbs_fb(w0,w1);

	sha1_me_generalised(m, 5);

	for (int t = 17; t < lastQ; ++t)
	{
		sha1_step(t, Q, m);
	}

	for (int t = 14; t <= lastQ; ++t)
	{
		uint32_t okm = isQokay(t,Q);
		if (okm != 0)
		{
			VERIFY_ERROR("Q" << t << " bad !" << hex << okm << dec);
			ok = false;
		}
	}

	// verify msgbitrelations
	// [1200] verify message bitrelations
	for (unsigned r = 0; r < msgbitrels16_size; ++r)
	{
		uint32_t w = msgbitrels16[r][16];
		for (unsigned t = mainblockoffset; t < mainblockoffset + 16; ++t)
		{
			w ^= m[t] & msgbitrels16[r][t - mainblockoffset];
		}
		if (0 != (hc::hw(w) & 1))
		{
			std::cout << "bitrelation " << r << " is not satisfied!" << std::endl;
			print_convert_msgbitrel(msgbitrels16[r], 1, 4);
			return false;
		}
	}

	return ok;
}


bool verify_Q27_Q30(int block, size_t read_idx, int lastQ, uint32_t w0, uint32_t w1, uint32_t q112boom)
{
	bool ok = true;
	using namespace host;

	uint32_t m[80];
	uint32_t Q[85];

	size_t ext_idx = unpack_idx(w0,w1);
	size_t base_idx = q20_solutions_buf[block].get<10>(ext_idx);

	Q[QOFF + 12] = base_solutions_buf.get<0>(base_idx);
	Q[QOFF + 13] = base_solutions_buf.get<1>(base_idx);
	Q[QOFF + 14] = base_solutions_buf.get<2>(base_idx);

	Q[QOFF + 15] = q20_solutions_buf[block].get<0>(ext_idx);
	Q[QOFF + 16] = q20_solutions_buf[block].get<1>(ext_idx);
	Q[QOFF + 17] = q20_solutions_buf[block].get<2>(ext_idx);

	m[ 5] = base_solutions_buf.get< 6>(base_idx);
	m[ 6] = base_solutions_buf.get< 7>(base_idx);
	m[ 7] = base_solutions_buf.get< 8>(base_idx);
	m[ 8] = base_solutions_buf.get< 9>(base_idx);
	m[ 9] = base_solutions_buf.get<10>(base_idx);
	m[10] = base_solutions_buf.get<11>(base_idx);
	m[11] = base_solutions_buf.get<12>(base_idx);
	m[12] = base_solutions_buf.get<13>(base_idx);
	m[13] = base_solutions_buf.get<14>(base_idx);

	m[14]        = q20_solutions_buf[block].get<3>(ext_idx);
	m[15]        = q20_solutions_buf[block].get<4>(ext_idx);
	m[16]        = q20_solutions_buf[block].get<5>(ext_idx);
	m[17]        = q20_solutions_buf[block].get<6>(ext_idx);
	m[18]        = q20_solutions_buf[block].get<7>(ext_idx);
	m[19]        = q20_solutions_buf[block].get<8>(ext_idx);
	m[20]        = q20_solutions_buf[block].get<9>(ext_idx);

	m[17] ^= unpack_w17ext_nbs(w0,w1);
	m[18] ^= unpack_w18_nbs(w0,w1);
	m[19] &= ~W19NBPACKM;
	m[19] ^= unpack_w19_nbs_fb(w0,w1);
	m[20] &= ~W20NBPACKM;
	m[20] ^= unpack_w20_nbs_fb(w0,w1);

	uint32_t q11boom = q112boom & Q11BOOMS;
	uint32_t q12boom = (q112boom>>16) & Q12BOOMS;
	m[10] ^= q11boom;
	m[11] ^= q11boom<<5;
	m[14] ^= (q11boom>>2)&(1<<6);
	m[15] ^= q11boom>>2;

	m[11] ^= q12boom;
	m[12] ^= q12boom<<5;
	m[16] ^= q12boom>>2;

	sha1_me_generalised(m, 5);

	for (int t = 17; t < lastQ; ++t)
	{
		sha1_step(t, Q, m);
	}

	for (int t = 14; t <= lastQ; ++t)
	{
		uint32_t okm = isQokay(t,Q);
		if (okm != 0)
		{
			VERIFY_ERROR("Q" << t << " bad !" << hex << okm << dec);
			ok = false;
		}
	}
	// verify msgbitrelations
	// [1200] verify message bitrelations
	for (unsigned r = 0; r < msgbitrels16_size; ++r)
	{
		uint32_t w = msgbitrels16[r][16];
		for (unsigned t = mainblockoffset; t < mainblockoffset + 16; ++t)
		{
			w ^= m[t] & msgbitrels16[r][t - mainblockoffset];
		}
		if (0 != (hc::hw(w) & 1))
		{
			std::cout << "bitrelation " << r << " is not satisfied!" << std::endl;
			print_convert_msgbitrel(msgbitrels16[r], 1, 4);
			return false;
		}
	}
	return ok;
}
























void verify_step_computations(int cuda_blocks)
{
	for (unsigned block = 0; block < cuda_blocks; ++block)
	{
		cout << "======== Verifying block " << block << endl;
/*
		cout << "Base solutions left: " << (base_solutions_ctl.write_idx - base_solutions_ctl.read_idx) << setfill(' ') << endl;
		size_t q18checked = 0, q18ok = 0;
		uint32_t q18count = q18_solutions_ctl_bu[block].write_idx - q18_solutions_ctl_bu[block].read_idx;
		cout << q18_solutions_ctl_bu[block].read_idx << " " << q18_solutions_ctl_bu[block].write_idx << " " << q18count << endl;
		for (uint32_t i = q18_solutions_ctl_bu[block].read_idx; i != q18_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q18_solutions_buf[block].get<0>(i);
			uint32_t w1 = q18_solutions_buf[block].get<1>(i);
			if (verify_Q18_Q19(block, i, 18, w0, w1))
				++q18ok;
			++q18checked;
			if (i - q18_solutions_ctl_bu[block].read_idx > q18_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10) << q18checked << " out of " << setw(10) << q18count << " Q18 solutions: " << q18ok << " OK" << endl;

		size_t q19checked = 0, q19ok = 0;
		uint32_t q19count = q19_solutions_ctl_bu[block].write_idx - q19_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q19_solutions_ctl_bu[block].read_idx; i != q19_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q19_solutions_buf[block].get<0>(i);
			uint32_t w1 = q19_solutions_buf[block].get<1>(i);
			if (verify_Q18_Q19(block, i, 19, w0, w1))
				++q19ok;
			++q19checked;
			if (i - q19_solutions_ctl_bu[block].read_idx > q19_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q19checked << " out of " << setw(10) << q19count  << " Q19 solutions: " << q19ok << " OK" << endl;
*/
		size_t q20checked = 0, q20ok = 0;
		uint32_t q20count = q20_solutions_ctl_bu[block].write_idx - q20_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q20_solutions_ctl_bu[block].read_idx; i != q20_solutions_ctl_bu[block].write_idx; ++i)
		{
			if (verify_Q20(block, i))
				++q20ok;
			++q20checked;
			if (i - q20_solutions_ctl_bu[block].read_idx > q20_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q20checked << " out of " << setw(10) << q20count  << " Q20 solutions: " << q20ok << " OK" << endl;

		size_t q21checked = 0, q21ok = 0;
		uint32_t q21count = q21_solutions_ctl_bu[block].write_idx - q21_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q21_solutions_ctl_bu[block].read_idx; i != q21_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q21_solutions_buf[block].get<0>(i);
			uint32_t w1 = q21_solutions_buf[block].get<1>(i);
			if (verify_Q21_Q26(block, i, 21, w0, w1))
				++q21ok;
			++q21checked;
			if (i - q21_solutions_ctl_bu[block].read_idx > q21_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q21checked << " out of " << setw(10) << q21count  << " Q21 solutions: " << q21ok << " OK" << endl;

		size_t q22checked = 0, q22ok = 0;
		uint32_t q22count = q22_solutions_ctl_bu[block].write_idx - q22_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q22_solutions_ctl_bu[block].read_idx; i != q22_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q22_solutions_buf[block].get<0>(i);
			uint32_t w1 = q22_solutions_buf[block].get<1>(i);
			if (verify_Q21_Q26(block, i, 22, w0, w1))
				++q22ok;
			++q22checked;
			if (i - q22_solutions_ctl_bu[block].read_idx > q22_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q22checked << " out of " << setw(10) << q22count  << " Q22 solutions: " << q22ok << " OK" << endl;

		size_t q23checked = 0, q23ok = 0;
		uint32_t q23count = q23_solutions_ctl_bu[block].write_idx - q23_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q23_solutions_ctl_bu[block].read_idx; i != q23_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q23_solutions_buf[block].get<0>(i);
			uint32_t w1 = q23_solutions_buf[block].get<1>(i);
			if (verify_Q21_Q26(block, i, 23, w0, w1))
				++q23ok;
			++q23checked;
			if (i - q23_solutions_ctl_bu[block].read_idx > q23_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q23checked << " out of " << setw(10) << q23count  << " Q23 solutions: " << q23ok << " OK" << endl;

		size_t q26checked = 0, q26ok = 0;
		uint32_t q26count = q26_solutions_ctl_bu[block].write_idx - q26_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q26_solutions_ctl_bu[block].read_idx; i != q26_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q26_solutions_buf[block].get<0>(i);
			uint32_t w1 = q26_solutions_buf[block].get<1>(i);
			if (verify_Q21_Q26(block, i, 26, w0, w1))
				++q26ok;
			++q26checked;
			if (i - q26_solutions_ctl_bu[block].read_idx > q26_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q26checked << " out of " << setw(10) << q26count  << " Q26 solutions: " << q26ok << " OK" << endl;

		size_t q28checked = 0, q28ok = 0;
		uint32_t q28count = q28_solutions_ctl_bu[block].write_idx - q28_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q28_solutions_ctl_bu[block].read_idx; i != q28_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q28_solutions_buf[block].get<0>(i);
			uint32_t w1 = q28_solutions_buf[block].get<1>(i);
			uint32_t w2 = q28_solutions_buf[block].get<2>(i);
			if (verify_Q27_Q30(block, i, 28, w0, w1, w2))
				++q28ok;
			++q28checked;
			if (i - q28_solutions_ctl_bu[block].read_idx > q28_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q28checked << " out of " << setw(10) << q28count  << " Q28 solutions: " << q28ok << " OK" << endl;

		size_t q30checked = 0, q30ok = 0;
		uint32_t q30count = q30_solutions_ctl_bu[block].write_idx - q30_solutions_ctl_bu[block].read_idx;
		for (uint32_t i = q30_solutions_ctl_bu[block].read_idx; i != q30_solutions_ctl_bu[block].write_idx; ++i)
		{
			uint32_t w0 = q30_solutions_buf[block].get<0>(i);
			uint32_t w1 = q30_solutions_buf[block].get<1>(i);
			uint32_t w2 = q30_solutions_buf[block].get<2>(i);
			if (verify_Q27_Q30(block, i, 30, w0, w1, w2))
				++q30ok;
			++q30checked;
			if (i - q30_solutions_ctl_bu[block].read_idx > q30_solutions_ctl_bu[block].size) break;
		}
		cout << "Verified " << setw(10)  << q30checked << " out of " << setw(10) << q30count  << " Q30 solutions: " << q30ok << " OK" << endl;

	}
}

#endif // VERIFY_GPU_RESULTS
