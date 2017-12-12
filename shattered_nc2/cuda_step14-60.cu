/*****
  Copyright (C) 2017 Pierre Karpman, INRIA France/Nanyang Technological University Singapore (-2016), CWI (2016/2017), L'Universite Grenoble Alpes (2017-)
            (C) 2017 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1_gpu_nearcollisionattacks source-code and released under the MIT License
*****/

/************ TODO TODO **********\
Add CUDA popcount to types.hpp to be cleaner ?
step_extend_Q33 early abort
move neutral bits without bug
\************ TODO TODO *********/


//// main prepocessor flags

// enables managed cyclic buffers and CPU verification of GPU results
//#define DEBUG1
// disabling temporary buffer will force writes to directly go to main buffer
//#define DISABLE_TMP_BUF // TODO was disabled. Was there a reason?

// enable performance counters
//#define USE_PERF_COUNTERS
// PERFORMANCE COUNTERS ARE NOW WORKING PROPERLY AND HAVE VERY SMALL OVERHEAD


#ifndef DEBUG1
#define BLOCKS 30
#define THREADS_PER_BLOCK 512
#define DEBUG_BREAK
#else
#define BLOCKS 2
#define THREADS_PER_BLOCK 512
#define DEBUG_BREAK //break;
#endif










#include "main.hpp"
#include "neutral_bits_packing.hpp"

#include "cuda_cyclicbuffer.hpp"
#include "sha1detail.hpp"

#include <timer.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace hashclash;
using namespace std;

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

// definition of cyclic buffer for 2^16 32-word elems: basesol: Q1,..,Q16,m0,...,m15 [uses CAS, as it's only written by the host]
typedef cyclic_buffer_cas_t< Q14BUFSIZE, uint32_t, 32, cyclic_buffer_control_cas_t< Q14BUFSIZE > > buffer_q14sol_t;
typedef buffer_q14sol_t::control_t control_q14sol_t;

// definition of cyclic buffer for 2^20 1-word elems
typedef cyclic_buffer_mask_t< (1<<20), uint32_t, 1, cyclic_buffer_control_mask_t< (1<<20) >, 1 > buffer_20_1_t; // used for block-specific buffers: fencetype = block-wide
typedef buffer_20_1_t::control_t control_20_1_t;

typedef cyclic_buffer_cas_t< (1<<20), uint32_t, 1, cyclic_buffer_control_cas_t< (1<<20) >, 2 > gl_buffer_20_1_t; // used for global buffers: fencetype = gpu-wide
typedef gl_buffer_20_1_t::control_t gl_control_20_1_t;

// definition of cyclic buffer for 2^20 2-word elems
typedef cyclic_buffer_mask_t< (1<<20), uint32_t, 2, cyclic_buffer_control_mask_t< (1<<20) >, 1 > buffer_20_2_t; // used for block-specific buffers: fencetype = block-wide
typedef buffer_20_2_t::control_t control_20_2_t;

typedef cyclic_buffer_cas_t< (1<<20), uint32_t, 2, cyclic_buffer_control_cas_t< (1<<20) >, 2 > gl_buffer_20_2_t; // used for global buffers: fencetype = gpu-wide
typedef gl_buffer_20_2_t::control_t gl_control_20_2_t;

// definition of cyclic buffer for 2^20 12-word elems
typedef cyclic_buffer_mask_t< (1<<19), uint32_t, 12, cyclic_buffer_control_mask_t< (1<<19) >, 1 > buffer_extbasesol20_t; // used for block-specific buffers: fencetype = block-wide
typedef buffer_extbasesol20_t::control_t control_extbasesol20_t;

// definition of cyclic buffer for 2^10 21-word elems: sol: Q36,..,Q40,m24,...,m39
// definition of cyclic buffer for 2^10 21-word elems: sol: Q56,..,Q60,m44,...,m59
typedef cyclic_buffer_cas_t< (1<<16), uint32_t, 21, cyclic_buffer_control_cas_t< (1<<16) >, 1 > buffer_sol_t; // used for block-specific buffers: fencetype = block-wide
typedef buffer_sol_t::control_t control_sol_t;
typedef cyclic_buffer_cas_t< (1<<10), uint32_t, 21, cyclic_buffer_control_cas_t< (1<<10) >, 2 > gl_buffer_sol_t; // used for global buffers: fencetype = gpu-wide
typedef gl_buffer_sol_t::control_t gl_control_sol_t;


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
MANAGED     __device__ buffer_q14sol_t   q14_solutions_buf;
__managed__ __device__ control_q14sol_t  q14_solutions_ctl;  // always managed to detect when it's empty
#define Q14SOLBUF                        q14_solutions_buf
#define Q14SOLCTL                        q14_solutions_ctl

MANAGED  __device__ buffer_20_1_t   q14aux_solutions_buf[BLOCKS];
         __shared__ control_20_1_t  q14aux_solutions_ctl;
MANAGED2 __device__ control_20_1_t  q14aux_solutions_ctl_bu[BLOCKS];
#define Q14AUXBUF                   q14aux_solutions_buf[blockIdx.x]
#define Q14AUXCTL                   q14aux_solutions_ctl

MANAGED  __device__ buffer_extbasesol20_t   q15_solutions_buf[BLOCKS];
         __shared__ control_extbasesol20_t  q15_solutions_ctl;
MANAGED2 __device__ control_extbasesol20_t  q15_solutions_ctl_bu[BLOCKS];
#define Q15SOLBUF                           q15_solutions_buf[blockIdx.x]
#define Q15SOLCTL                           q15_solutions_ctl

MANAGED  __device__ buffer_20_2_t           q16_solutions_buf[BLOCKS];
         __shared__ control_20_2_t          q16_solutions_ctl;
MANAGED2 __device__ control_20_2_t          q16_solutions_ctl_bu [BLOCKS];
#define Q16SOLBUF                           q16_solutions_buf    [blockIdx.x]
#define Q16SOLCTL                           q16_solutions_ctl

MANAGED  __device__ buffer_20_2_t           q17_solutions_buf[BLOCKS];
         __shared__ control_20_2_t          q17_solutions_ctl;
MANAGED2 __device__ control_20_2_t          q17_solutions_ctl_bu [BLOCKS];
#define Q17SOLBUF                           q17_solutions_buf    [blockIdx.x]
#define Q17SOLCTL                           q17_solutions_ctl

MANAGED  __device__ buffer_extbasesol20_t   q18_solutions_buf[BLOCKS];
         __shared__ control_extbasesol20_t  q18_solutions_ctl;
MANAGED2 __device__ control_extbasesol20_t  q18_solutions_ctl_bu[BLOCKS];
#define Q18SOLBUF                           q18_solutions_buf[blockIdx.x]
#define Q18SOLCTL                           q18_solutions_ctl

MANAGED  __device__ buffer_20_1_t   q19_solutions_buf[BLOCKS];
         __shared__ control_20_1_t  q19_solutions_ctl;
MANAGED2 __device__ control_20_1_t  q19_solutions_ctl_bu[BLOCKS];
#define Q19SOLBUF                   q19_solutions_buf[blockIdx.x]
#define Q19SOLCTL                   q19_solutions_ctl

MANAGED  __device__ buffer_20_1_t   q21_solutions_buf[BLOCKS];
         __shared__ control_20_1_t  q21_solutions_ctl;
MANAGED2 __device__ control_20_1_t  q21_solutions_ctl_bu[BLOCKS];
#define Q21SOLBUF                   q21_solutions_buf[blockIdx.x]
#define Q21SOLCTL                   q21_solutions_ctl

MANAGED  __device__ buffer_20_2_t           q23_solutions_buf[BLOCKS];
         __shared__ control_20_2_t          q23_solutions_ctl;
MANAGED2 __device__ control_20_2_t          q23_solutions_ctl_bu [BLOCKS];
#define Q23SOLBUF                           q23_solutions_buf    [blockIdx.x]
#define Q23SOLCTL                           q23_solutions_ctl

MANAGED __device__ buffer_sol_t  q26_solutions_buf[BLOCKS];
__shared__ control_sol_t q26_solutions_ctl;
MANAGED2 __device__ control_sol_t q26_solutions_ctl_bu [BLOCKS];
#define Q26SOLBUF                 q26_solutions_buf    [blockIdx.x]
#define Q26SOLCTL                 q26_solutions_ctl

MANAGED __device__ buffer_sol_t  q33_solutions_buf[BLOCKS];
__shared__ control_sol_t q33_solutions_ctl;
MANAGED2 __device__ control_sol_t q33_solutions_ctl_bu [BLOCKS];
#define Q33SOLBUF                 q33_solutions_buf    [blockIdx.x]
#define Q33SOLCTL                 q33_solutions_ctl

/*
MANAGED __device__ buffer_sol_t  q53_solutions_buf[BLOCKS];
__shared__ control_sol_t q53_solutions_ctl;
MANAGED2 __device__ control_sol_t q53_solutions_ctl_bu [BLOCKS];
#define Q53SOLBUF                 q53_solutions_buf    [blockIdx.x]
#define Q53SOLCTL                 q53_solutions_ctl
*/
__managed__ __device__ gl_buffer_sol_t  q53_solutions_buf;
__managed__ __device__ gl_control_sol_t q53_solutions_ctl;
#define Q53SOLBUF  q53_solutions_buf
#define Q53SOLCTL  q53_solutions_ctl

//__managed__ __device__ gl_buffer_sol_t  collision_candidates_buf;
//__managed__ __device__ gl_control_sol_t collision_candidates_ctl;
//#define COLLCANDIDATEBUF  collision_candidates_buf
//#define COLLCANDIDATECTL  collision_candidates_ctl





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
#include "tables_org.hpp"
using namespace tbl_org;
}

namespace dev {
#define TABLE_PREFIX __constant__
#include "tables_org.hpp"
using namespace tbl_org;
}


/* *** SHA1 FUNCTIONS **********************************
 */
__host__ __device__ inline uint32_t sha1_round1(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d, const uint32_t e, const uint32_t m)
{
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

	return rotate_left(a,5) + sha1_f3(b, rotate_left(c,30), rotate_left(d,30)) + rotate_left(e,30) + m + 0x8F1BBCDC;
}

__host__ __device__ inline uint32_t sha1_round4(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d, const uint32_t e, const uint32_t m)
{

	return rotate_left(a,5) + sha1_f4(b, rotate_left(c,30), rotate_left(d,30)) + rotate_left(e,30) + m + 0xCA62C1D6;
}

__host__ __device__ inline uint32_t sha1_mess(uint32_t m_3, uint32_t m_8, uint32_t m_14, uint32_t m_16)
{
	return rotate_left(m_3 ^ m_8 ^ m_14 ^ m_16, 1);
}

#define NEXT_NB(a,mask) { (a) -= 1; (a) &= mask;}

__device__ void stepQ14aux(uint32_t q14idx)
{
	PERF_START_COUNTER(14);
	using namespace dev;

	for (unsigned i = 0; i < (1<<5); ++i)
		Q14AUXBUF.write(Q14AUXCTL, true, (q14idx | (i << 20)));

	PERF_STOP_COUNTER(14);
}

__device__ void stepQ15(uint32_t q14auxidx)
{
	PERF_START_COUNTER(15);
	using namespace dev;

	q14auxidx = Q14AUXBUF.get<0>(q14auxidx);
	uint32_t q14idx = q14auxidx & 0xFFFFF;
	uint32_t fixnbs = (q14auxidx >> 14) & 0x7C0;

	uint32_t q07 = Q14SOLBUF.get<15+7>(q14idx);
	uint32_t q08 = Q14SOLBUF.get<15+8>(q14idx);
	uint32_t q09 = Q14SOLBUF.get<15+9>(q14idx);
	uint32_t q10 = Q14SOLBUF.get<15+10>(q14idx);
	uint32_t q11 = Q14SOLBUF.get<15+11>(q14idx);
	uint32_t q12 = Q14SOLBUF.get<15+12>(q14idx);
	uint32_t q13 = Q14SOLBUF.get<15+13>(q14idx);
	uint32_t q14 = Q14SOLBUF.get<15+14>(q14idx);
	uint32_t q15;

	uint32_t m10 = Q14SOLBUF.get<10>(q14idx);
	uint32_t m11 = Q14SOLBUF.get<11>(q14idx);
	uint32_t m12 = Q14SOLBUF.get<12>(q14idx);
	uint32_t m13 = Q14SOLBUF.get<13>(q14idx);
	uint32_t m14 = Q14SOLBUF.get<14>(q14idx);
	uint32_t m15 = Q14SOLBUF.get<15>(q14idx);

	uint32_t oldq12 = q12;
	uint32_t oldq13 = q13;
	uint32_t oldq14 = q14;

	// preset fixed neutral bits W10[6..10]
	m10 ^= fixnbs;
	q11 += fixnbs;

	// W11 neutral bits
	for (int W11t9b  = 0; W11t9b  < 2; ++W11t9b)  { m11 ^= 0x200;
	for (int W11t10b = 0; W11t10b < 2; ++W11t10b) { m11 ^= 0x400; m14 ^= 0x800;
	for (int W11t11b = 0; W11t11b < 2; ++W11t11b) { m11 ^= 0x800;
	for (int W11t12b = 0; W11t12b < 2; ++W11t12b) { m11 ^= 0x1000; m14 ^= 0x800;
	for (int W11t13b = 0; W11t13b < 2; ++W11t13b) { m11 ^= 0x2000; m13 ^= 0x1304000; m14 ^= 0x352800; m15 ^= 0x4003c000;
	for (int W11t14b = 0; W11t14b < 2; ++W11t14b) { m11 ^= 0x4000; m15 ^= 0x800;
	for (int W11t15b = 0; W11t15b < 2; ++W11t15b) { m11 ^= 0x8000; m13 ^= 0x1300000; m14 ^= 0x352800; m15 ^= 0x4003c000;

	q12 = sha1_round1(q11, q10, q09, q08, q07, m11);
	bool valid_sol11 = (0 == ((oldq12 ^ q12) & Qcondmask[QOFF + 12]));

	// W12 neutral bits
	for (int W12t2b  = 0; W12t2b  < 2; ++W12t2b)  { m12 ^= 0x4; m13 ^= 0x80100000; m14 ^= 0x40fe0002; m15 ^= 0x17e8004;
	for (int W12t14b = 0; W12t14b < 2; ++W12t14b) { m12 ^= 0x4000; m14 ^= 0x2800;
	for (int W12t15b = 0; W12t15b < 2; ++W12t15b) { m12 ^= 0x8000; m14 ^= 0x5000; m15 ^= 0x800;
	for (int W12t16b = 0; W12t16b < 2; ++W12t16b) { m12 ^= 0x10000; m13 ^= 0x1304000; m14 ^= 0x358800; m15 ^= 0x4003c800;
	for (int W12t17b = 0; W12t17b < 2; ++W12t17b) { m12 ^= 0x20000; m14 ^= 0x14000; m15 ^= 0x2000;
	for (int W12t18b = 0; W12t18b < 2; ++W12t18b) { m12 ^= 0x40000; m13 ^= 0x4000; m14 ^= 0x29000; m15 ^= 0x6800;
	for (int W12t19b = 0; W12t19b < 2; ++W12t19b) { m12 ^= 0x80000; m14 ^= 0x53800; m15 ^= 0xc800;
	for (int W12t20b = 0; W12t20b < 2; ++W12t20b) { m12 ^= 0x100000; m13 ^= 0x1284000; m14 ^= 0x2d6800; m15 ^= 0x4130e804;

	q13 = sha1_round1(q12, q11, q10, q09, q08, m12);
	bool valid_sol12 = valid_sol11 & (0 == ((oldq13 ^ q13) & Qcondmask[QOFF + 13]));

	// W13 neutral bits
	for (int W13t12b = 0; W13t12b < 2; ++W13t12b) { m13 ^= 0x1305000; m14 ^= 0x352800; m15 ^= 0x4003c000;
	for (int W13t16b = 0; W13t16b < 2; ++W13t16b) { m13 ^= 0x190000; m14 ^= 0x120000; m15 ^= 0x1328804;

	q14 = sha1_round1(q13, q12, q11, q10, q09, m13);
	bool valid_sol13 = valid_sol12 & (0 == ((oldq14 ^ q14) & Qcondmask[QOFF + 14]));

	q15 = sha1_round1(q14, q13, q12, q11, q10, m14);

	uint32_t q15nessies = Qset1mask[QOFF + 15] 	^ (Qprevmask[QOFF + 15] & q14)
//												^ (Qprevrmask [QOFF + 15] & rotate_left(q14, 30))
//												^ (Qprev2rmask[QOFF + 15] & rotate_left(q13, 30))
	;
	valid_sol13 &= (0 == ((q15 ^ q15nessies) & Qcondmask[QOFF + 15]));

	Q15SOLBUF.write(Q15SOLCTL, valid_sol13, q11, q12, q13, q14, q15, m10, m11, m12, m13, m14, m15, q14idx);

	}}
	}}}}}}}}
	}}}}}}}

	PERF_STOP_COUNTER(15);
}

__device__ void stepQ16(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(16);

	using namespace dev;

	const uint32_t q14idx = Q15SOLBUF.get<11>(thread_rd_idx);

	uint32_t q09 = Q14SOLBUF.get<15+9>(q14idx);
	uint32_t q10 = Q14SOLBUF.get<15+10>(q14idx);
	uint32_t q11 = Q15SOLBUF.get<0>(thread_rd_idx);
	uint32_t q12 = Q15SOLBUF.get<1>(thread_rd_idx);
	uint32_t q13 = Q15SOLBUF.get<2>(thread_rd_idx);
	uint32_t q14 = Q15SOLBUF.get<3>(thread_rd_idx);
	uint32_t q15 = Q15SOLBUF.get<4>(thread_rd_idx);
	uint32_t m11 = Q15SOLBUF.get<6>(thread_rd_idx);
	uint32_t m12 = Q15SOLBUF.get<7>(thread_rd_idx);
	uint32_t m13 = Q15SOLBUF.get<8>(thread_rd_idx);
	uint32_t m14 = Q15SOLBUF.get<9>(thread_rd_idx);
	uint32_t m15 = Q15SOLBUF.get<10>(thread_rd_idx);

	uint32_t oldq13 = q13;

	uint32_t w11_q16_nb = 0;
	for (unsigned l = 0; l < (1<<2); ++l)
	{
		NEXT_NB(w11_q16_nb, W11NBQ16M);

		m11 &= ~W11NBQ16M;
		m11 |= w11_q16_nb;

		q12 += w11_q16_nb;
		q13 += rotate_left(w11_q16_nb, 5);

		uint32_t w12_q16_nb = 0;
		for (unsigned k = 0; k < (1<<5); ++k)
		{
			NEXT_NB(w12_q16_nb, W12NBQ16M);

			m12 &= ~W12NBQ16M;
			m12 |= w12_q16_nb;

			// first flips
			uint32_t m14f;
		   	m14f  = (w12_q16_nb & 0x00003000) >> 1;
			m14f ^= (w12_q16_nb & 0x00002000) >> 2;
			m14  ^= m14f;

			q13 += w12_q16_nb;

			uint32_t newq14 = sha1_round1(q13, q12, q11, q10, q09, m13);

			bool valid_sol1 = (0 == ((oldq13 ^ q13) & Qcondmask[QOFF + 13]));
			valid_sol1 &= (0 == ((newq14 ^ q14) & Qcondmask[QOFF + 14]));

			uint32_t w13_q16_nb = 0;
			for (unsigned j = 0; j < (1<<2); ++j)
			{
				NEXT_NB(w13_q16_nb, W13NBQ16M);

				m13 &= ~W13NBQ16M;
				m13 |= w13_q16_nb;

				// second flips
				m14 ^= (__popc(w13_q16_nb) & 1) << 12;

				newq14 += w13_q16_nb; // nothing can go wrong for newq14 at this point

				uint32_t newq15 = sha1_round1(newq14, q13, q12, q11, q10, m14);
				bool valid_sol2 = valid_sol1 & (0 == ((newq15 ^ q15) & Qcondmask[QOFF + 15]));

				uint32_t newq16 = sha1_round1(newq15, newq14, q13, q12, q11, m15);

				uint32_t q16nessies = Qset1mask[QOFF + 16] 	^ (Qprevmask[QOFF + 16] & newq15)
//															^ (Qprevrmask [QOFF + 16] & rotate_left(newq15, 30))
//															^ (Qprev2rmask[QOFF + 16] & rotate_left(newq14, 30))
				;
				valid_sol2 &= (0 == ((newq16 ^ q16nessies) & Qcondmask[QOFF + 16]));

				uint32_t sol_val_0 = pack_q16q17_sol0(thread_rd_idx, m11, m12, m13, m14);
				uint32_t sol_val_1 = pack_q16q17_sol1(thread_rd_idx, m11, m12, m13, m14);
				WARP_TMP_BUF.write2(valid_sol2, sol_val_0, sol_val_1, Q16SOLBUF, Q16SOLCTL);

				m14 ^= (__popc(w13_q16_nb) & 1) << 12;

				newq14 -= w13_q16_nb;
			}

			m14 ^= m14f;

			q13 -= w12_q16_nb;
		}

		q13 -= rotate_left(w11_q16_nb, 5);
		q12 -= w11_q16_nb;
	}

	WARP_TMP_BUF.flush2(Q16SOLBUF, Q16SOLCTL);
	PERF_STOP_COUNTER(16);
}


__device__ void stepQ17(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(17);

	using namespace dev;

	uint32_t q16_sol0 = Q16SOLBUF.get<0>(thread_rd_idx);
	uint32_t q16_sol1 = Q16SOLBUF.get<1>(thread_rd_idx);

	const uint32_t q15idx = unpack_q15idx(q16_sol0, q16_sol1);
	const uint32_t q14idx = Q15SOLBUF.get<11>(q15idx);

	uint32_t q09 = Q14SOLBUF.get<15+9>(q14idx);
	uint32_t q10 = Q14SOLBUF.get<15+10>(q14idx);

	uint32_t q11 = Q15SOLBUF.get<0>(q15idx);
	uint32_t q12 = Q15SOLBUF.get<1>(q15idx);
	uint32_t q13 = Q15SOLBUF.get<2>(q15idx);
	uint32_t m11 = Q15SOLBUF.get<6>(q15idx);
	uint32_t m12 = Q15SOLBUF.get<7>(q15idx);
	uint32_t m13 = Q15SOLBUF.get<8>(q15idx);
	uint32_t m14 = Q15SOLBUF.get<9>(q15idx);
	uint32_t m15 = Q15SOLBUF.get<10>(q15idx);

	{
		uint32_t w11_sol_nb = unpack_w11_nbs(q16_sol0, q16_sol1);
		uint32_t w12_sol_nb = unpack_w12_nbs(q16_sol0, q16_sol1);
		uint32_t w13_sol_nb = unpack_w13_nbs(q16_sol0, q16_sol1);
		uint32_t w14_sol_fl = unpack_w14_fls(q16_sol0, q16_sol1); // Warning: those are *explicit* values, it's not a flip mask

		m11 |= w11_sol_nb;
		m12 |= w12_sol_nb;
		m13 |= w13_sol_nb;
		m14 &= W14FLQ16M;
		m14 |= w14_sol_fl;

		q12 += w11_sol_nb;
		q13 += w12_sol_nb + rotate_left(w11_sol_nb, 5);
	}

	uint32_t m16;
	{
		uint32_t m00 = Q14SOLBUF.get<0>(q14idx);
		uint32_t m02 = Q14SOLBUF.get<2>(q14idx);
		uint32_t m08 = Q14SOLBUF.get<8>(q14idx);
		m16 = sha1_mess(0, m08, m02, m00);
	}

	uint32_t q14    = sha1_round1(q13, q12, q11, q10, q09, m13);
	uint32_t oldq15 = sha1_round1(q14, q13, q12, q11, q10, m14);
	uint32_t oldq16 = sha1_round1(oldq15, q14, q13, q12, q11, m15);

	uint32_t w12_q17_nb = 0;
	for (unsigned l = 0; l < (1<<4); ++l)
	{
		NEXT_NB(w12_q17_nb, W12NBQ17M);

		m12 &= ~W12NBQ17M;
		m12 |= w12_q17_nb;

		q13 += w12_q17_nb;
		q14 += rotate_left(w12_q17_nb, 5);

		uint32_t w13_q17_nb = 0;
		for (unsigned k = 0; k < (1<<3); ++k)
		{
			NEXT_NB(w13_q17_nb, W13NBQ17M);

			m13 &= ~W13NBQ17M;
			m13 |= w13_q17_nb;

			q14 += w13_q17_nb;
			m16 ^= rotate_left(m13, 1);

			uint32_t q15 = sha1_round1(q14, q13, q12, q11, q10, m14); // could be partially precomputed before (not clear it'd be better)
			uint32_t q16 = sha1_round1(q15, q14, q13, q12, q11, m15);
			uint32_t q17 = sha1_round1(q16, q15, q14, q13, q12, m16);

			bool valid_sol = (0 == ((oldq15 ^ q15) & Qcondmask[QOFF + 15]));
			valid_sol &= (0 == ((oldq16 ^ q16) & Qcondmask[QOFF + 16]));

			uint32_t q17nessies = Qset1mask[QOFF + 17] 	^ (Qprevmask[QOFF + 17] & q16)
//														^ (Qprevrmask [QOFF + 17] & rotate_left(q16, 30))
//														^ (Qprev2rmask[QOFF + 17] & rotate_left(q15, 30))
			;
			valid_sol &= (0 == ((q17 ^ q17nessies) & Qcondmask[QOFF + 17]));

			uint32_t sol_val_0 = pack_q16q17_sol0(q15idx, m11, m12, m13, m14);
			uint32_t sol_val_1 = pack_q16q17_sol1(q15idx, m11, m12, m13, m14);
			WARP_TMP_BUF.write2(valid_sol, sol_val_0, sol_val_1, Q17SOLBUF, Q17SOLCTL);

			m16 ^= rotate_left(m13, 1);
			q14 -= w13_q17_nb;
		}

		q13 -= w12_q17_nb;
		q14 -= rotate_left(w12_q17_nb, 5);
	}

	WARP_TMP_BUF.flush2(Q17SOLBUF, Q17SOLCTL);
	PERF_STOP_COUNTER(17);
}


__device__ void stepQ18(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(18);

	using namespace dev;

	uint32_t q17_sol0 = Q17SOLBUF.get<0>(thread_rd_idx);
	uint32_t q17_sol1 = Q17SOLBUF.get<1>(thread_rd_idx);

	const uint32_t q15idx = unpack_q15idx(q17_sol0, q17_sol1);
	const uint32_t q14idx = Q15SOLBUF.get<11>(q15idx);

	uint32_t q09 = Q14SOLBUF.get<15+9>(q14idx);
	uint32_t q10 = Q14SOLBUF.get<15+10>(q14idx);
	uint32_t m01 = Q14SOLBUF.get<1>(q14idx);
	uint32_t m03 = Q14SOLBUF.get<3>(q14idx);
	uint32_t m09 = Q14SOLBUF.get<9>(q14idx);

	uint32_t q11 = Q15SOLBUF.get<0>(q15idx);
	uint32_t q12 = Q15SOLBUF.get<1>(q15idx);
	uint32_t q13 = Q15SOLBUF.get<2>(q15idx);
	uint32_t m11 = Q15SOLBUF.get<6>(q15idx);
	uint32_t m12 = Q15SOLBUF.get<7>(q15idx);
	uint32_t m13 = Q15SOLBUF.get<8>(q15idx);
	uint32_t m14 = Q15SOLBUF.get<9>(q15idx);
	uint32_t m15 = Q15SOLBUF.get<10>(q15idx);

	{
		uint32_t w11_sol_nb = unpack_w11_nbs(q17_sol0, q17_sol1);
		uint32_t w12_sol_nb = unpack_w12_nbs(q17_sol0, q17_sol1);
		uint32_t w13_sol_nb = unpack_w13_nbs(q17_sol0, q17_sol1);
		uint32_t w14_sol_fl = unpack_w14_fls(q17_sol0, q17_sol1); // Warning: those are *explicit* values, it's not a flip mask

		m11 |= w11_sol_nb;
		m12 |= w12_sol_nb;
		m13 |= w13_sol_nb;
		m14 &= W14FLQ16M;
		m14 |= w14_sol_fl;

		q12 += w11_sol_nb;
		q13 += w12_sol_nb + rotate_left(w11_sol_nb, 5);
	}

	uint32_t m16;
	{
		uint32_t m00 = Q14SOLBUF.get<0>(q14idx);
		uint32_t m02 = Q14SOLBUF.get<2>(q14idx);
		uint32_t m08 = Q14SOLBUF.get<8>(q14idx);

		m16 = sha1_mess(m13, m08, m02, m00);
	}

	uint32_t q14    = sha1_round1(q13, q12, q11, q10, q09, m13);
	uint32_t q15    = sha1_round1(q14, q13, q12, q11, q10, m14);
	uint32_t oldq16 = sha1_round1(q15, q14, q13, q12, q11, m15);
	uint32_t oldq17 = sha1_round1(oldq16, q15, q14, q13, q12, m16);

	uint32_t w13_q18_nb = 0;
	for (unsigned l = 0; l < (1<<5); ++l)
	{
		NEXT_NB(w13_q18_nb, W13NBQ18M);

		m13 &= ~W13NBQ18M;
		m13 |= w13_q18_nb;

		q14 += w13_q18_nb;

		// flip
		m14 ^= (w13_q18_nb & 0x00000200) << 2;

		m16 ^= rotate_left(w13_q18_nb, 1);

		uint32_t m17 = sha1_mess(m14, m09, m03, m01); // can be precomputed, but it's not clear it's better

		q15 = sha1_round1(q14, q13, q12, q11, q10, m14); // to get the flip into account
		uint32_t q16 = sha1_round1(q15, q14, q13, q12, q11, m15);

		// the unique w14_q18_nb inline; val = 0
		{
			bool valid_sol = (0 == ((oldq16 ^ q16) & Qcondmask[QOFF + 16]));

			uint32_t q17 = sha1_round1(q16, q15, q14, q13, q12, m16);
			valid_sol &= (0 == ((oldq17 ^ q17) & Qcondmask[QOFF + 17]));

			uint32_t q18 = sha1_round1(q17, q16, q15, q14, q13, m17);

			uint32_t q18nessies = Qset1mask[QOFF + 18] 	// ^ (Qprevmask[QOFF + 18] & q17)
//														^ (Qprevrmask [QOFF + 18] & rotate_left(q17, 30))
//														^ (Qprev2rmask[QOFF + 18] & rotate_left(q16, 30))
			;
			valid_sol &= (0 == ((q18 ^ q18nessies) & Qcondmask[QOFF + 18]));

			Q18SOLBUF.write(Q18SOLCTL, valid_sol, q12, q13, q14, q15, q16, q17, m11, m12, m13, m14, m15, q15idx);
		}
		// the unique w14_q18_nb inline; val = 1
		{
			m17 ^= rotate_left((uint32_t)W14NBQ18M, 1);

			q15 += W14NBQ18M;
			q16 += rotate_left((uint32_t)W14NBQ18M, 5);

			bool valid_sol = (0 == ((oldq16 ^ q16) & Qcondmask[QOFF + 16]));

			uint32_t q17 = sha1_round1(q16, q15, q14, q13, q12, m16);
			valid_sol &= (0 == ((oldq17 ^ q17) & Qcondmask[QOFF + 17]));

			uint32_t q18 = sha1_round1(q17, q16, q15, q14, q13, m17);

			uint32_t q18nessies = Qset1mask[QOFF + 18] 	// ^ (Qprevmask[QOFF + 18] & q17)
//														^ (Qprevrmask [QOFF + 18] & rotate_left(q17, 30))
//														^ (Qprev2rmask[QOFF + 18] & rotate_left(q16, 30))
			;
			valid_sol &= (0 == ((q18 ^ q18nessies) & Qcondmask[QOFF + 18]));

			Q18SOLBUF.write(Q18SOLCTL, valid_sol, q12, q13, q14, q15, q16, q17, m11, m12, m13, m14 ^ W14NBQ18M, m15, q15idx);

			// no need to undo m17, as it's fully recomputed at every iteration anyway
			// no need to undo q15, as it's fully recomputed at every iteration anyway
			// no need to undo q16, as it's fully recomputed at every iteration anyway
		}

		q14 -= w13_q18_nb;

		// flip
		m14 ^= (w13_q18_nb & 0x00000200) << 2;

		m16 ^= rotate_left(w13_q18_nb, 1);
	}

	PERF_STOP_COUNTER(18);
}


__device__ void stepQ19(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(19);

	using namespace dev;

	const uint32_t q15idx = Q18SOLBUF.get<11>(thread_rd_idx);
	const uint32_t q14idx = Q15SOLBUF.get<11>(q15idx);

	uint32_t q12 = Q18SOLBUF.get<0>(thread_rd_idx);
	uint32_t q13 = Q18SOLBUF.get<1>(thread_rd_idx);
	uint32_t q14 = Q18SOLBUF.get<2>(thread_rd_idx);
	uint32_t q15 = Q18SOLBUF.get<3>(thread_rd_idx);
	uint32_t q16 = Q18SOLBUF.get<4>(thread_rd_idx);
	uint32_t q17 = Q18SOLBUF.get<5>(thread_rd_idx);


	uint32_t m14 = Q18SOLBUF.get<9>(thread_rd_idx);
	uint32_t m15 = Q18SOLBUF.get<10>(thread_rd_idx);

	uint32_t m16;
	uint32_t m17;
	uint32_t m18;
	{
		uint32_t m00 = Q14SOLBUF.get<0>(q14idx);
		uint32_t m01 = Q14SOLBUF.get<1>(q14idx);
		uint32_t m02 = Q14SOLBUF.get<2>(q14idx);
		uint32_t m03 = Q14SOLBUF.get<3>(q14idx);
		uint32_t m04 = Q14SOLBUF.get<4>(q14idx);
		uint32_t m08 = Q14SOLBUF.get<8>(q14idx);
		uint32_t m09 = Q14SOLBUF.get<9>(q14idx);
		uint32_t m10 = Q15SOLBUF.get<5>(q15idx);
		uint32_t m13 = Q18SOLBUF.get<8>(thread_rd_idx);

		m16 = sha1_mess(m13, m08, m02, m00);
		m17 = sha1_mess(0, m09, m03, m01);
		m18 = sha1_mess(0, m10, m04, m02);
	}

	uint32_t oldq16 = q16;

	uint32_t w14_q19_nb = 0;
	for (unsigned l = 0; l < (1<<3); ++l)
	{
		NEXT_NB(w14_q19_nb, W14NBQ19M);

		m14 &= ~W14NBQ19M;
		m14 |= w14_q19_nb;

		m17 ^= rotate_left(m14, 1);

		q15 += w14_q19_nb;
		q16 += rotate_left(w14_q19_nb, 5);

		bool valid_sol1 = (0 == ((oldq16 ^ q16) & Qcondmask[QOFF + 16]));

		uint32_t w15_q19_nb = 0;
		for (unsigned k = 0; k < (1<<2); ++k)
		{
			NEXT_NB(w15_q19_nb, W15NBQ19M);

			m15 &= ~W15NBQ19M;
			m15 |= w15_q19_nb;

			m18 ^= rotate_left(m15, 1);

			q16 += w15_q19_nb;

			uint32_t newq17 = sha1_round1(q16, q15, q14, q13, q12, m16);
			bool valid_sol2 = valid_sol1 & (0 == ((newq17 ^ q17) & Qcondmask[QOFF + 17]));

			uint32_t newq18 = sha1_round1(newq17, q16, q15, q14, q13, m17);
			uint32_t q18nessies = Qset1mask[QOFF + 18] 	//^ (Qprevmask[QOFF + 18] & newq17)
//														^ (Qprevrmask [QOFF + 18] & rotate_left(newq17, 30))
//														^ (Qprev2rmask[QOFF + 18] & rotate_left(q16, 30))
			;
			valid_sol2 &= (0 == ((newq18 ^ q18nessies) & Qcondmask[QOFF + 18]));

			uint32_t newq19 = sha1_round1(newq18, newq17, q16, q15, q14, m18);
			uint32_t q19nessies = Qset1mask[QOFF + 19] 	//^ (Qprevmask[QOFF + 19] & newq18)
//														^ (Qprevrmask [QOFF + 19] & rotate_left(newq18, 30))
														^ (Qprev2rmask[QOFF + 19] & rotate_left(newq17, 30))
			;
			valid_sol2 &= (0 == ((newq19 ^ q19nessies) & Qcondmask[QOFF + 19]));

			uint32_t sol_val_0 = pack_q19q21_sol0(thread_rd_idx, m14, m15);
			WARP_TMP_BUF.write1(valid_sol2, sol_val_0, Q19SOLBUF, Q19SOLCTL);

			m18 ^= rotate_left(m15, 1);

			q16 -= w15_q19_nb;
		}

		m17 ^= rotate_left(m14, 1);

		q15 -= w14_q19_nb;
		q16 -= rotate_left(w14_q19_nb, 5);
	}

	WARP_TMP_BUF.flush1(Q19SOLBUF, Q19SOLCTL);
	PERF_STOP_COUNTER(19);
}

__device__ void stepQ201(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(21);

	using namespace dev;

	uint32_t q19_sol0 = Q19SOLBUF.get<0>(thread_rd_idx);

	const uint32_t q18idx = unpack_q18idx(q19_sol0);
	const uint32_t q15idx = Q18SOLBUF.get<11>(q18idx);
	const uint32_t q14idx = Q15SOLBUF.get<11>(q15idx);

	uint32_t q12 = Q18SOLBUF.get<0>(q18idx);
	uint32_t q13 = Q18SOLBUF.get<1>(q18idx);
	uint32_t q14 = Q18SOLBUF.get<2>(q18idx);
	uint32_t q15 = Q18SOLBUF.get<3>(q18idx);
	uint32_t q16 = Q18SOLBUF.get<4>(q18idx);
	uint32_t q17 = Q18SOLBUF.get<5>(q18idx);

	uint32_t m14 = Q18SOLBUF.get<9>(q18idx);
	uint32_t m15 = Q18SOLBUF.get<10>(q18idx);

	{
		uint32_t w14_sol_nb = unpack_w14_nbs(q19_sol0);
		uint32_t w15_sol_nb = unpack_w15_nbs(q19_sol0);

		m14 |= w14_sol_nb;
		m15 |= w15_sol_nb;

		q15 += w14_sol_nb;
		q16 += rotate_left(w14_sol_nb, 5);
		q16 += w15_sol_nb;
	}

	uint32_t m16;
	uint32_t m17;
	uint32_t m18;
	uint32_t m19;
	uint32_t m20;
	{
		uint32_t m00 = Q14SOLBUF.get<0>(q14idx);
		uint32_t m01 = Q14SOLBUF.get<1>(q14idx);
		uint32_t m02 = Q14SOLBUF.get<2>(q14idx);
		uint32_t m03 = Q14SOLBUF.get<3>(q14idx);
		uint32_t m04 = Q14SOLBUF.get<4>(q14idx);
		uint32_t m05 = Q14SOLBUF.get<5>(q14idx);
		uint32_t m06 = Q14SOLBUF.get<6>(q14idx);
		uint32_t m08 = Q14SOLBUF.get<8>(q14idx);
		uint32_t m09 = Q14SOLBUF.get<9>(q14idx);
		uint32_t m10 = Q15SOLBUF.get<5>(q15idx);
		uint32_t m11 = Q18SOLBUF.get<6>(q18idx);
		uint32_t m12 = Q18SOLBUF.get<7>(q18idx);
		uint32_t m13 = Q18SOLBUF.get<8>(q18idx);

		m16 = sha1_mess(m13, m08, m02, m00);
		m17 = sha1_mess(0, m09, m03, m01);
		m18 = sha1_mess(0, m10, m04, m02);
		m19 = sha1_mess(m16, m11, m05, m03);
		m20 = sha1_mess(0, m12, m06, m04);
	}

	uint32_t w14_q20_nb = 0;
	for (unsigned l = 0; l < (1<<2); ++l)
	{
		NEXT_NB(w14_q20_nb, W14NBQ20M);

		m14 &= ~W14NBQ20M;
		m14 |= w14_q20_nb;

		q15 += w14_q20_nb;
		q16 += rotate_left(w14_q20_nb, 5);

		m17 ^= rotate_left(m14, 1);
		m20 ^= rotate_left(m17, 1);

		uint32_t w15_q20_nb = 0;
		for (unsigned k = 0; k < (1<<5); ++k)
		{
			NEXT_NB(w15_q20_nb, W15NBQ20M);

			m15 &= ~W15NBQ20M;
			m15 |= w15_q20_nb;

			m18 ^= rotate_left(m15, 1);

			q16 += w15_q20_nb;

			uint32_t newq17 = sha1_round1(q16, q15, q14, q13, q12, m16);
			bool valid_sol = (0 == ((newq17 ^ q17) & Qcondmask[QOFF + 17]));

			uint32_t newq18 = sha1_round1(newq17, q16, q15, q14, q13, m17);
			uint32_t q18nessies = Qset1mask[QOFF + 18] 	//^ (Qprevmask[QOFF + 18] & newq17)
//														^ (Qprevrmask [QOFF + 18] & rotate_left(newq17, 30))
//														^ (Qprev2rmask[QOFF + 18] & rotate_left(q16, 30))
			;
			valid_sol &= (0 == ((newq18 ^ q18nessies) & Qcondmask[QOFF + 18]));

			uint32_t newq19 = sha1_round1(newq18, newq17, q16, q15, q14, m18);
			uint32_t q19nessies = Qset1mask[QOFF + 19] 	//^ (Qprevmask[QOFF + 19] & newq18)
//														^ (Qprevrmask [QOFF + 19] & rotate_left(newq18, 30))
														^ (Qprev2rmask[QOFF + 19] & rotate_left(newq17, 30))
			;
			valid_sol &= (0 == ((newq19 ^ q19nessies) & Qcondmask[QOFF + 19]));

			uint32_t newq20 = sha1_round1(newq19, newq18, newq17, q16, q15, m19);
			uint32_t q20nessies = Qset1mask[QOFF + 20] 	//^ (Qprevmask[QOFF + 20] & newq19)
														^ (Qprevrmask [QOFF + 20] & rotate_left(newq19, 30))
//														^ (Qprev2rmask[QOFF + 20] & rotate_left(newq18, 30))
			;
			valid_sol &= (0 == ((newq20 ^ q20nessies) & Qcondmask[QOFF + 20]));

			uint32_t newq21 = sha1_round2(newq20, newq19, newq18, newq17, q16, m20);
			uint32_t q21nessies = Qset1mask[QOFF + 21] 	^ (Qprevmask[QOFF + 21] & newq20)
//													^ (Qprevrmask [QOFF + 21] & rotate_left(newq20, 30))
//													^ (Qprev2rmask[QOFF + 21] & rotate_left(newq19, 30))
			;
			valid_sol &= (0 == ((newq21 ^ q21nessies) & Qcondmask[QOFF + 21]));

			uint32_t sol_val_0 = pack_q19q21_sol0(q18idx, m14, m15);
			WARP_TMP_BUF.write1(valid_sol, sol_val_0, Q21SOLBUF, Q21SOLCTL);

			m18 ^= rotate_left(m15, 1);

			q16 -= w15_q20_nb;
		}

		q15 -= w14_q20_nb;
		q16 -= rotate_left(w14_q20_nb, 5);

		m20 ^= rotate_left(m17, 1);
		m17 ^= rotate_left(m14, 1);
	}

	WARP_TMP_BUF.flush1(Q21SOLBUF, Q21SOLCTL);
	PERF_STOP_COUNTER(21);
}

__device__ void stepQ23(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(23);

	using namespace dev;

	uint32_t q21_sol0 = Q21SOLBUF.get<0>(thread_rd_idx);

	const uint32_t q18idx = unpack_q18idx(q21_sol0);
	const uint32_t q15idx = Q18SOLBUF.get<11>(q18idx);
	const uint32_t q14idx = Q15SOLBUF.get<11>(q15idx);

	uint32_t q12 = Q18SOLBUF.get<0>(q18idx);
	uint32_t q13 = Q18SOLBUF.get<1>(q18idx);
	uint32_t q14 = Q18SOLBUF.get<2>(q18idx);
	uint32_t q15 = Q18SOLBUF.get<3>(q18idx);
	uint32_t q16 = Q18SOLBUF.get<4>(q18idx);
	uint32_t q17;
	uint32_t q18;
	uint32_t q19;
	uint32_t q20;
	uint32_t q21;
	uint32_t q22;
	uint32_t q23;

	uint32_t m00 = Q14SOLBUF.get<0>(q14idx);
	uint32_t m01 = Q14SOLBUF.get<1>(q14idx);
	uint32_t m02 = Q14SOLBUF.get<2>(q14idx);
	uint32_t m03 = Q14SOLBUF.get<3>(q14idx);
	uint32_t m04 = Q14SOLBUF.get<4>(q14idx);
	uint32_t m05 = Q14SOLBUF.get<5>(q14idx);
	uint32_t m06 = Q14SOLBUF.get<6>(q14idx);
	uint32_t m07 = Q14SOLBUF.get<7>(q14idx);
	uint32_t m08 = Q14SOLBUF.get<8>(q14idx);
	uint32_t m09 = Q14SOLBUF.get<9>(q14idx);
	uint32_t m10 = Q15SOLBUF.get<5>(q15idx);
	uint32_t m11 = Q18SOLBUF.get<6>(q18idx);
	uint32_t m12 = Q18SOLBUF.get<7>(q18idx);
	uint32_t m13 = Q18SOLBUF.get<8>(q18idx);
	uint32_t m14 = Q18SOLBUF.get<9>(q18idx);
	uint32_t m15 = Q18SOLBUF.get<10>(q18idx);
	uint32_t m16;
	uint32_t m17;
	uint32_t m18;
	uint32_t m19;
	uint32_t m20;
	uint32_t m21;
	uint32_t m22;
	uint32_t m23;

	{
		uint32_t w14_sol_nb = unpack_w14_nbs(q21_sol0);
		uint32_t w15_sol_nb = unpack_w15_nbs(q21_sol0);

		m14 |= w14_sol_nb;
		m15 |= w15_sol_nb;

		q15 += w14_sol_nb;
		q16 += rotate_left(w14_sol_nb, 5);
		q16 += w15_sol_nb;
	}

	// booms change: m09, m10, m14 => m17(m09,m14), m18(m10), m22(m14)

	m16 = sha1_mess(m13, m08, m02, m00);

	uint32_t q10_bo = 0;
	for (unsigned m = 0; m < (1<<1); ++m)
	{
		NEXT_NB(q10_bo, Q10BOOMS);

		// TODO optim
		m09 ^= q10_bo;
	   	m10 ^= rotate_left(q10_bo, 5);
		m14 ^= rotate_right(q10_bo, 2);

		// TODO optim
		m17 = sha1_mess(m14, m09, m03, m01);
		m18 = sha1_mess(m15, m10, m04, m02);
		m19 = sha1_mess(m16, m11, m05, m03);
		m20 = sha1_mess(m17, m12, m06, m04);
		m21 = sha1_mess(m18, m13, m07, m05);
		m22 = sha1_mess(m19, m14, m08, m06);
		m23 = sha1_mess(m20, m15, m09, m07);

		q17 = sha1_round1(q16, q15, q14, q13, q12, m16);
		q18 = sha1_round1(q17, q16, q15, q14, q13, m17);

		q19 = sha1_round1(q18, q17, q16, q15, q14, m18);
		uint32_t q19nessies = Qset1mask[QOFF + 19] 	//^ (Qprevmask[QOFF + 19] & q18)
//														^ (Qprevrmask [QOFF + 19] & rotate_left(q18, 30))
														^ (Qprev2rmask[QOFF + 19] & rotate_left(q17, 30))
		;
		bool valid_sol = 0 == ((q19 ^ q19nessies) & Qcondmask[QOFF + 19]);

		q20 = sha1_round1(q19, q18, q17, q16, q15, m19);
		uint32_t q20nessies = Qset1mask[QOFF + 20] 	//^ (Qprevmask[QOFF + 20] & q19)
														^ (Qprevrmask [QOFF + 20] & rotate_left(q19, 30))
//														^ (Qprev2rmask[QOFF + 20] & rotate_left(q18, 30))
		;
		valid_sol &= 0 == ((q20 ^ q20nessies) & Qcondmask[QOFF + 20]);

		q21 = sha1_round2(q20, q19, q18, q17, q16, m20);
		uint32_t q21nessies = Qset1mask[QOFF + 21] 	^ (Qprevmask[QOFF + 21] & q20)
//													^ (Qprevrmask [QOFF + 21] & rotate_left(q20, 30))
//													^ (Qprev2rmask[QOFF + 21] & rotate_left(q19, 30))
		;
		valid_sol &= 0 == ((q21 ^ q21nessies) & Qcondmask[QOFF + 21]);

		q22 = sha1_round2(q21, q20, q19, q18, q17, m21);
		uint32_t q22nessies = Qset1mask[QOFF + 22] 	// ^ (Qprevmask[QOFF + 22] & q21)
													^ (Qprevrmask [QOFF + 22] & rotate_left(q21, 30))
//													^ (Qprev2rmask[QOFF + 22] & rotate_left(q20, 30))
		;
		q22nessies ^= (m23 & 0x08000000); // message-dependent condition
		valid_sol &= 0 == ((q22 ^ q22nessies) & Qcondmask[QOFF + 22]);

		q23 = sha1_round2(q22, q21, q20, q19, q18, m22);
		uint32_t q23nessies = Qset1mask[QOFF + 23] 	^ (Qprevmask[QOFF + 23] & q22)
//													^ (Qprevrmask [QOFF + 23] & rotate_left(q22, 30))
													^ (Qprev2rmask[QOFF + 23] & rotate_left(q21, 30))
		;
		valid_sol &= 0 == ((q23 ^ q23nessies) & Qcondmask[QOFF + 23]);

		WARP_TMP_BUF.write2(valid_sol, q21_sol0, q10_bo, Q23SOLBUF, Q23SOLCTL);

		m09 ^= q10_bo;
	   	m10 ^= rotate_left(q10_bo, 5);
		m14 ^= rotate_right(q10_bo, 2);
	}

	WARP_TMP_BUF.flush2(Q23SOLBUF, Q23SOLCTL);
	PERF_STOP_COUNTER(23);
}

__device__ void stepQ26(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(26);

	using namespace dev;

	uint32_t q23_sol0 = Q23SOLBUF.get<0>(thread_rd_idx);
	uint32_t q10_bo = Q23SOLBUF.get<1>(thread_rd_idx);

	const uint32_t q18idx = unpack_q18idx(q23_sol0);
	const uint32_t q15idx = Q18SOLBUF.get<11>(q18idx);
	const uint32_t q14idx = Q15SOLBUF.get<11>(q15idx);

	uint32_t q12 = Q18SOLBUF.get<0>(q18idx);
	uint32_t q13 = Q18SOLBUF.get<1>(q18idx);
	uint32_t q14 = Q18SOLBUF.get<2>(q18idx);
	uint32_t q15 = Q18SOLBUF.get<3>(q18idx);
	uint32_t q16 = Q18SOLBUF.get<4>(q18idx);
	uint32_t q17;
	uint32_t q18;
	uint32_t q19;
	uint32_t q20;
	uint32_t q21;
	uint32_t q22;
	uint32_t q23;
	uint32_t q24;
	uint32_t q25;
	uint32_t q26;

	uint32_t m00 = Q14SOLBUF.get<0>(q14idx);
	uint32_t m01 = Q14SOLBUF.get<1>(q14idx);
	uint32_t m02 = Q14SOLBUF.get<2>(q14idx);
	uint32_t m03 = Q14SOLBUF.get<3>(q14idx);
	uint32_t m04 = Q14SOLBUF.get<4>(q14idx);
	uint32_t m05 = Q14SOLBUF.get<5>(q14idx);
	uint32_t m06 = Q14SOLBUF.get<6>(q14idx);
	uint32_t m07 = Q14SOLBUF.get<7>(q14idx);
	uint32_t m08 = Q14SOLBUF.get<8>(q14idx);
	uint32_t m09 = Q14SOLBUF.get<9>(q14idx);
	uint32_t m10 = Q15SOLBUF.get<5>(q15idx);
	uint32_t m11 = Q18SOLBUF.get<6>(q18idx);
	uint32_t m12 = Q18SOLBUF.get<7>(q18idx);
	uint32_t m13 = Q18SOLBUF.get<8>(q18idx);
	uint32_t m14 = Q18SOLBUF.get<9>(q18idx);
	uint32_t m15 = Q18SOLBUF.get<10>(q18idx);
	uint32_t m16;
	uint32_t m17;
	uint32_t m18;
	uint32_t m19;
	uint32_t m20;
	uint32_t m21;
	uint32_t m22;
	uint32_t m23;
	uint32_t m24;
	uint32_t m25;
	uint32_t m26;
	uint32_t m27;

	{
		uint32_t w14_sol_nb = unpack_w14_nbs(q23_sol0);
		uint32_t w15_sol_nb = unpack_w15_nbs(q23_sol0);

		m14 |= w14_sol_nb;
		m15 |= w15_sol_nb;

		q15 += w14_sol_nb;
		q16 += rotate_left(w14_sol_nb, 5);
		q16 += w15_sol_nb;
	}

	m09 ^= q10_bo;
	m10 ^= rotate_left(q10_bo, 5);
	m14 ^= rotate_right(q10_bo, 2);

	// booms change: m06, m07, m11 => m19(m11), m20(m06), m21(m07), m22(m06), m23(m07), m24(m08)

	m16 = sha1_mess(m13, m08, m02, m00);
	m17 = sha1_mess(m14, m09, m03, m01);
	m18 = sha1_mess(m15, m10, m04, m02);

	uint32_t q07_bo = 0;
	for (unsigned m = 0; m < (1<<2); ++m)
	{
		NEXT_NB(q07_bo, Q07BOOMS);

		m06 ^= q07_bo;
		m07 ^= rotate_left(q07_bo, 5);
		m11 ^= rotate_right(q07_bo, 2);

		// TODO optim
		m19 = sha1_mess(m16, m11, m05, m03);
		m20 = sha1_mess(m17, m12, m06, m04);
		m21 = sha1_mess(m18, m13, m07, m05);
		m22 = sha1_mess(m19, m14, m08, m06);
		m23 = sha1_mess(m20, m15, m09, m07);
		m24 = sha1_mess(m21, m16, m10, m08);
		m25 = sha1_mess(m22, m17, m11, m09);
		m26 = sha1_mess(m23, m18, m12, m10);
		m27 = sha1_mess(m24, m19, m13, m11);

		q17 = sha1_round1(q16, q15, q14, q13, q12, m16);
		q18 = sha1_round1(q17, q16, q15, q14, q13, m17);
		q19 = sha1_round1(q18, q17, q16, q15, q14, m18);
		q20 = sha1_round1(q19, q18, q17, q16, q15, m19);
		q21 = sha1_round2(q20, q19, q18, q17, q16, m20);

		q22 = sha1_round2(q21, q20, q19, q18, q17, m21);
		uint32_t q22nessies = Qset1mask[QOFF + 22] 	// ^ (Qprevmask[QOFF + 22] & q21)
			^ (Qprevrmask [QOFF + 22] & rotate_left(q21, 30))
			//													^ (Qprev2rmask[QOFF + 22] & rotate_left(q20, 30))
			;
		q22nessies ^= (m23 & 0x08000000); // message-dependent condition
		bool valid_sol = 0 == ((q22 ^ q22nessies) & Qcondmask[QOFF + 22]);

		q23 = sha1_round2(q22, q21, q20, q19, q18, m22);
		uint32_t q23nessies = Qset1mask[QOFF + 23] 	^ (Qprevmask[QOFF + 23] & q22)
			//													^ (Qprevrmask [QOFF + 23] & rotate_left(q22, 30))
			^ (Qprev2rmask[QOFF + 23] & rotate_left(q21, 30))
			;
		valid_sol &= 0 == ((q23 ^ q23nessies) & Qcondmask[QOFF + 23]);

		q24 = sha1_round2(q23, q22, q21, q20, q19, m23);
		uint32_t q24nessies = Qset1mask[QOFF + 24] 	//^ (Qprevmask[QOFF + 24] & q23)
			^ (Qprevrmask [QOFF + 24] & rotate_left(q23, 30))
			//													^ (Qprev2rmask[QOFF + 24] & rotate_left(q22, 30))
			;
		q24nessies ^= (m25 & 0x08000000); // message-dependent condition
		valid_sol &= 0 == ((q24 ^ q24nessies) & Qcondmask[QOFF + 24]);

		q25 = sha1_round2(q24, q23, q22, q21, q20, m24);
		uint32_t q25nessies = Qset1mask[QOFF + 25] 	//^ (Qprevmask[QOFF + 25] & q24)
			//													^ (Qprevrmask [QOFF + 25] & rotate_left(q24, 30))
			^ (Qprev2rmask[QOFF + 25] & rotate_left(q23, 30))
			;
		q25nessies ^= (m25 & 0x08000000)<<1; // message-dependent condition
		valid_sol &= 0 == ((q25 ^ q25nessies) & Qcondmask[QOFF + 25]);

		// TEH XTRA COND
		if ((q24 ^ m23) & 0x40000000)
		{
			valid_sol &= 0 == ((q24 ^ rotate_left(m23, 1)) & 0x80000000);
		}
		else
		{
			valid_sol &= 0 == ((q24 ^ rotate_left(q25, 2) ^ rotate_left(m23, 4)) & 0x80000000);
		}

		q26 = sha1_round2(q25, q24, q23, q22, q21, m25);
		uint32_t q26nessies = Qset1mask[QOFF + 26] 	//^ (Qprevmask[QOFF + 26] & q25)
			^ (Qprevrmask [QOFF + 26] & rotate_left(q25, 30))
			//													^ (Qprev2rmask[QOFF + 26] & rotate_left(q24, 30))
			;
		q26nessies ^= (m27 & 0x08000000); // message-dependent condition
		valid_sol &= 0 == ((q26 ^ q26nessies) & Qcondmask[QOFF + 26]);

		Q26SOLBUF.write(Q26SOLCTL, valid_sol, q19, q20, q21, q22, q23,
												m12, m13, m14, m15, m16, m17, m18, m19,
												m20, m21, m22, m23, m24, m25, m26, m27);


		m06 ^= q07_bo;
		m07 ^= rotate_left(q07_bo, 5);
		m11 ^= rotate_right(q07_bo, 2);
	}

	PERF_STOP_COUNTER(26);
}

__device__ void step_extend_Q33(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(33);
	using namespace dev;


	uint32_t m17 = Q26SOLBUF.get<10>(thread_rd_idx);
	uint32_t m18 = Q26SOLBUF.get<11>(thread_rd_idx);
	uint32_t m19 = Q26SOLBUF.get<12>(thread_rd_idx);
	uint32_t m20 = Q26SOLBUF.get<13>(thread_rd_idx);
	uint32_t m21 = Q26SOLBUF.get<14>(thread_rd_idx);
	uint32_t m22 = Q26SOLBUF.get<15>(thread_rd_idx);
	uint32_t m23 = Q26SOLBUF.get<16>(thread_rd_idx);
	uint32_t m24 = Q26SOLBUF.get<17>(thread_rd_idx);
	uint32_t m25 = Q26SOLBUF.get<18>(thread_rd_idx);
	uint32_t m26 = Q26SOLBUF.get<19>(thread_rd_idx);
	uint32_t m27 = Q26SOLBUF.get<20>(thread_rd_idx);
	uint32_t m28, m29, m30, m31, m32;

	{
		uint32_t m12 = Q26SOLBUF.get<5>(thread_rd_idx);
		uint32_t m13 = Q26SOLBUF.get<6>(thread_rd_idx);
		uint32_t m14 = Q26SOLBUF.get<7>(thread_rd_idx);
		uint32_t m15 = Q26SOLBUF.get<8>(thread_rd_idx);
		uint32_t m16 = Q26SOLBUF.get<9>(thread_rd_idx);

		m28 = sha1_mess(m25, m20, m14, m12);
		m29 = sha1_mess(m26, m21, m15, m13);
		m30 = sha1_mess(m27, m22, m16, m14);
		m31 = sha1_mess(m28, m23, m17, m15);
		m32 = sha1_mess(m29, m24, m18, m16);
	}

	uint32_t e = Q26SOLBUF.get<0>(thread_rd_idx); // q19
	uint32_t d = Q26SOLBUF.get<1>(thread_rd_idx); // q20
	uint32_t c = Q26SOLBUF.get<2>(thread_rd_idx); // q21
	uint32_t b = Q26SOLBUF.get<3>(thread_rd_idx); // q22
	uint32_t a = Q26SOLBUF.get<4>(thread_rd_idx); // q23
	uint32_t E = e + dQ[QOFF + 19];
	uint32_t D = d + dQ[QOFF + 20];
	uint32_t C = c + dQ[QOFF + 21];
	uint32_t B = b + dQ[QOFF + 22];
	uint32_t A = a + dQ[QOFF + 23];

	e = sha1_round2(a, b, c, d, e, m23);
	d = sha1_round2(e, a, b, c, d, m24);
	c = sha1_round2(d, e, a, b, c, m25);
	b = sha1_round2(c, d, e, a, b, m26);
	a = sha1_round2(b, c, d, e, a, m27);

	e = sha1_round2(a, b, c, d, e, m28);
	d = sha1_round2(e, a, b, c, d, m29);
	c = sha1_round2(d, e, a, b, c, m30);
	b = sha1_round2(c, d, e, a, b, m31);
	a = sha1_round2(b, c, d, e, a, m32);

	E = sha1_round2(A, B, C, D, E, m23 ^ DV_DW[23]);
	D = sha1_round2(E, A, B, C, D, m24 ^ DV_DW[24]);
	C = sha1_round2(D, E, A, B, C, m25 ^ DV_DW[25]);
	B = sha1_round2(C, D, E, A, B, m26 ^ DV_DW[26]);
	A = sha1_round2(B, C, D, E, A, m27 ^ DV_DW[27]);

	E = sha1_round2(A, B, C, D, E, m28 ^ DV_DW[28]);
	D = sha1_round2(E, A, B, C, D, m29 ^ DV_DW[29]);
	C = sha1_round2(D, E, A, B, C, m30 ^ DV_DW[30]);
	B = sha1_round2(C, D, E, A, B, m31 ^ DV_DW[31]);
	A = sha1_round2(B, C, D, E, A, m32 ^ DV_DW[32]);

	bool good33 = 0 == ((e^E)|(d^D)|(c^C)|(b^B)|(a^A));
	// sol: Q29,..,Q33,m17,...,m32
	Q33SOLBUF.write(Q33SOLCTL, good33, e, d, c, b, a,
									 m17, m18, m19, m20, m21, m22, m23, m24,
									 m25, m26, m27, m28, m29, m30, m31, m32);
	PERF_STOP_COUNTER(33);
}

__device__ void step_extend_Q53(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(53);
	using namespace dev;


	uint32_t m33, m34, m35, m36, m37, m38, m39, m40;
	uint32_t m41, m42, m43, m44, m45, m46, m47, m48;
	uint32_t m49, m50, m51, m52;

	{
		uint32_t m17 = Q33SOLBUF.get<5>(thread_rd_idx);
		uint32_t m18 = Q33SOLBUF.get<6>(thread_rd_idx);
		uint32_t m19 = Q33SOLBUF.get<7>(thread_rd_idx);
		uint32_t m20 = Q33SOLBUF.get<8>(thread_rd_idx);
		uint32_t m21 = Q33SOLBUF.get<9>(thread_rd_idx);
		uint32_t m22 = Q33SOLBUF.get<10>(thread_rd_idx);
		uint32_t m23 = Q33SOLBUF.get<11>(thread_rd_idx);
		uint32_t m24 = Q33SOLBUF.get<12>(thread_rd_idx);
		uint32_t m25 = Q33SOLBUF.get<13>(thread_rd_idx);
		uint32_t m26 = Q33SOLBUF.get<14>(thread_rd_idx);
		uint32_t m27 = Q33SOLBUF.get<15>(thread_rd_idx);
		uint32_t m28 = Q33SOLBUF.get<16>(thread_rd_idx);
		uint32_t m29 = Q33SOLBUF.get<17>(thread_rd_idx);
		uint32_t m30 = Q33SOLBUF.get<18>(thread_rd_idx);
		uint32_t m31 = Q33SOLBUF.get<19>(thread_rd_idx);
		uint32_t m32 = Q33SOLBUF.get<20>(thread_rd_idx);

		m33 = sha1_mess(m30, m25, m19, m17);
		m34 = sha1_mess(m31, m26, m20, m18);
		m35 = sha1_mess(m32, m27, m21, m19);
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
	}

	uint32_t e = Q33SOLBUF.get<0>(thread_rd_idx); // q29
	uint32_t d = Q33SOLBUF.get<1>(thread_rd_idx); // q30
	uint32_t c = Q33SOLBUF.get<2>(thread_rd_idx); // q31
	uint32_t b = Q33SOLBUF.get<3>(thread_rd_idx); // q32
	uint32_t a = Q33SOLBUF.get<4>(thread_rd_idx); // q33
	uint32_t E = e;
	uint32_t D = d;
	uint32_t C = c;
	uint32_t B = b;
	uint32_t A = a;

	e = sha1_round2(a, b, c, d, e, m33);
	d = sha1_round2(e, a, b, c, d, m34);
	c = sha1_round2(d, e, a, b, c, m35);
	b = sha1_round2(c, d, e, a, b, m36);
	a = sha1_round2(b, c, d, e, a, m37);

	e = sha1_round2(a, b, c, d, e, m38);
	d = sha1_round2(e, a, b, c, d, m39);
	c = sha1_round3(d, e, a, b, c, m40);
	b = sha1_round3(c, d, e, a, b, m41);
	a = sha1_round3(b, c, d, e, a, m42);

	e = sha1_round3(a, b, c, d, e, m43);
	d = sha1_round3(e, a, b, c, d, m44);
	c = sha1_round3(d, e, a, b, c, m45);
	b = sha1_round3(c, d, e, a, b, m46);
	a = sha1_round3(b, c, d, e, a, m47);

	e = sha1_round3(a, b, c, d, e, m48);
	d = sha1_round3(e, a, b, c, d, m49);
	c = sha1_round3(d, e, a, b, c, m50);
	b = sha1_round3(c, d, e, a, b, m51);
	a = sha1_round3(b, c, d, e, a, m52);

	E = sha1_round2(A, B, C, D, E, m33 ^ DV_DW[33]);
	D = sha1_round2(E, A, B, C, D, m34 ^ DV_DW[34]);
	C = sha1_round2(D, E, A, B, C, m35 ^ DV_DW[35]);
	B = sha1_round2(C, D, E, A, B, m36 ^ DV_DW[36]);
	A = sha1_round2(B, C, D, E, A, m37 ^ DV_DW[37]);

	E = sha1_round2(A, B, C, D, E, m38 ^ DV_DW[38]);
	D = sha1_round2(E, A, B, C, D, m39 ^ DV_DW[39]);
	C = sha1_round3(D, E, A, B, C, m40 ^ DV_DW[40]);
	B = sha1_round3(C, D, E, A, B, m41 ^ DV_DW[41]);
	A = sha1_round3(B, C, D, E, A, m42 ^ DV_DW[42]);

	E = sha1_round3(A, B, C, D, E, m43 ^ DV_DW[43]);
	D = sha1_round3(E, A, B, C, D, m44 ^ DV_DW[44]);
	C = sha1_round3(D, E, A, B, C, m45 ^ DV_DW[45]);
	B = sha1_round3(C, D, E, A, B, m46 ^ DV_DW[46]);
	A = sha1_round3(B, C, D, E, A, m47 ^ DV_DW[47]);

	E = sha1_round3(A, B, C, D, E, m48 ^ DV_DW[48]);
	D = sha1_round3(E, A, B, C, D, m49 ^ DV_DW[49]);
	C = sha1_round3(D, E, A, B, C, m50 ^ DV_DW[50]);
	B = sha1_round3(C, D, E, A, B, m51 ^ DV_DW[51]);
	A = sha1_round3(B, C, D, E, A, m52 ^ DV_DW[52]);

	bool good53 = 0 == ((e^E)|(d^D)|(c^C)|(b^B)|(a^A));
	// sol: Q49,..,Q53,m37,...,m52
	Q53SOLBUF.write(Q53SOLCTL, good53, e, d, c, b, a,
									 m37, m38, m39, m40, m41, m42, m43, m44,
									 m45, m46, m47, m48, m49, m50, m51, m52);
	PERF_STOP_COUNTER(53);
}

/*
__device__ void step_extend_Q61(uint32_t thread_rd_idx)
{
	PERF_START_COUNTER(61);
	using namespace dev;

	uint32_t m45 = Q53SOLBUF.get<13>(thread_rd_idx);
	uint32_t m46 = Q53SOLBUF.get<14>(thread_rd_idx);
	uint32_t m47 = Q53SOLBUF.get<15>(thread_rd_idx);
	uint32_t m48 = Q53SOLBUF.get<16>(thread_rd_idx);
	uint32_t m49 = Q53SOLBUF.get<17>(thread_rd_idx);
	uint32_t m50 = Q53SOLBUF.get<18>(thread_rd_idx);
	uint32_t m51 = Q53SOLBUF.get<19>(thread_rd_idx);
	uint32_t m52 = Q53SOLBUF.get<20>(thread_rd_idx);
	uint32_t m53, m54, m55, m56, m57, m58, m59, m60;

	{
		uint32_t m37 = Q53SOLBUF.get<5>(thread_rd_idx);
		uint32_t m38 = Q53SOLBUF.get<6>(thread_rd_idx);
		uint32_t m39 = Q53SOLBUF.get<7>(thread_rd_idx);
		uint32_t m40 = Q53SOLBUF.get<8>(thread_rd_idx);
		uint32_t m41 = Q53SOLBUF.get<9>(thread_rd_idx);
		uint32_t m42 = Q53SOLBUF.get<10>(thread_rd_idx);
		uint32_t m43 = Q53SOLBUF.get<11>(thread_rd_idx);
		uint32_t m44 = Q53SOLBUF.get<12>(thread_rd_idx);

		m53 = sha1_mess(m50, m45, m39, m37);
		m54 = sha1_mess(m51, m46, m40, m38);
		m55 = sha1_mess(m52, m47, m41, m39);
		m56 = sha1_mess(m53, m48, m42, m40);
		m57 = sha1_mess(m54, m49, m43, m41);
		m58 = sha1_mess(m55, m50, m44, m42);
		m59 = sha1_mess(m56, m51, m45, m43);
		m60 = sha1_mess(m57, m52, m46, m44);
	}

	uint32_t e = Q53SOLBUF.get<0>(thread_rd_idx); // q49
	uint32_t d = Q53SOLBUF.get<1>(thread_rd_idx); // q50
	uint32_t c = Q53SOLBUF.get<2>(thread_rd_idx); // q51
	uint32_t b = Q53SOLBUF.get<3>(thread_rd_idx); // q52
	uint32_t a = Q53SOLBUF.get<4>(thread_rd_idx); // q53
	uint32_t E = e;
	uint32_t D = d;
	uint32_t C = c;
	uint32_t B = b;
	uint32_t A = a;

	e = sha1_round3(a, b, c, d, e, m53);
	d = sha1_round3(e, a, b, c, d, m54);
	c = sha1_round3(d, e, a, b, c, m55);
	b = sha1_round3(c, d, e, a, b, m56);
	a = sha1_round3(b, c, d, e, a, m57);

	e = sha1_round3(a, b, c, d, e, m58);
	d = sha1_round3(e, a, b, c, d, m59);
	c = sha1_round4(d, e, a, b, c, m60);

	E = sha1_round3(A, B, C, D, E, m53 ^ DV_DW[53]);
	D = sha1_round3(E, A, B, C, D, m54 ^ DV_DW[54]);
	C = sha1_round3(D, E, A, B, C, m55 ^ DV_DW[55]);
	B = sha1_round3(C, D, E, A, B, m56 ^ DV_DW[56]);
	A = sha1_round3(B, C, D, E, A, m57 ^ DV_DW[57]);

	E = sha1_round3(A, B, C, D, E, m58 ^ DV_DW[58]);
	D = sha1_round3(E, A, B, C, D, m59 ^ DV_DW[59]);
	C = sha1_round4(D, E, A, B, C, m60 ^ DV_DW[60]);

	bool good61 = 0 == ((e^E)|(d^D)|(c^C)|(b^B)|(a^A));
	// sol: Q57,..,Q61,m45,...,m60
	COLLCANDIDATEBUF.write(COLLCANDIDATECTL, good61, b, a, e, d, c,
											 m45, m46, m47, m48, m49, m50, m51, m52,
											 m53, m54, m55, m56, m57, m58, m59, m60);
	PERF_STOP_COUNTER(61);
}
*/









// BACKUP CONTROLS ONLY IF THEY ARE IN SHARED (AND THUS BLOCK-SPECIFIC)
__device__ void backup_controls()
{
	__syncthreads();
	if (threadIdx.x == 0)
	{
		q14aux_solutions_ctl_bu[blockIdx.x] = Q14AUXCTL;
		q15_solutions_ctl_bu[blockIdx.x] = Q15SOLCTL;
		q16_solutions_ctl_bu[blockIdx.x] = Q16SOLCTL;
		q17_solutions_ctl_bu[blockIdx.x] = Q17SOLCTL;
		q18_solutions_ctl_bu[blockIdx.x] = Q18SOLCTL;
		q19_solutions_ctl_bu[blockIdx.x] = Q19SOLCTL;
		q21_solutions_ctl_bu[blockIdx.x] = Q21SOLCTL;
		q23_solutions_ctl_bu[blockIdx.x] = Q23SOLCTL;
		q26_solutions_ctl_bu[blockIdx.x] = Q26SOLCTL;
		q33_solutions_ctl_bu[blockIdx.x] = Q33SOLCTL;
		//q53_solutions_ctl_bu[blockIdx.x] = Q53SOLCTL;


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
		Q14AUXCTL = q14aux_solutions_ctl_bu[blockIdx.x];
		Q15SOLCTL = q15_solutions_ctl_bu[blockIdx.x];
		Q16SOLCTL = q16_solutions_ctl_bu[blockIdx.x];
		Q17SOLCTL = q17_solutions_ctl_bu[blockIdx.x];
		Q18SOLCTL = q18_solutions_ctl_bu[blockIdx.x];
		Q19SOLCTL = q19_solutions_ctl_bu[blockIdx.x];
		Q21SOLCTL = q21_solutions_ctl_bu[blockIdx.x];
		Q23SOLCTL = q23_solutions_ctl_bu[blockIdx.x];
		Q26SOLCTL = q26_solutions_ctl_bu[blockIdx.x];
		Q33SOLCTL = q33_solutions_ctl_bu[blockIdx.x];
		//Q53SOLCTL = q53_solutions_ctl_bu[blockIdx.x];

#ifdef USE_PERF_COUNTERS
		performance_restore();
#endif
	}
	__syncthreads();
}

__global__ void reset_buffers()
{
	// restore_controls(); // unnecessary

	Q14AUXBUF.reset(Q14AUXCTL);
	Q15SOLBUF.reset(Q15SOLCTL);
	Q16SOLBUF.reset(Q16SOLCTL);
	Q17SOLBUF.reset(Q17SOLCTL);
	Q18SOLBUF.reset(Q18SOLCTL);
	Q19SOLBUF.reset(Q19SOLCTL);
	Q21SOLBUF.reset(Q21SOLCTL);
	Q33SOLBUF.reset(Q33SOLCTL);
	//Q53SOLBUF.reset(Q53SOLCTL);

//	COLLCANDIDATEBUF.reset(COLLCANDIDATECTL);

#ifdef USE_PERF_COUNTERS
	performance_reset();
#endif
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

	do
	{

#if 0
		{
			uint32_t thidx = Q53SOLBUF.getreadidx(Q53SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				step_extend_Q61(thidx);
		//		break;
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q33SOLBUF.getreadidx(Q33SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				step_extend_Q53(thidx);
		//		break;
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
				step_extend_Q33(thidx);
		//		break;
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
				stepQ26(thidx);
		//		break;
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
				stepQ23(thidx);
		//		break;
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
				stepQ201(thidx);
		//		break;
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
		//		break;
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q17SOLBUF.getreadidx(Q17SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ18(thidx);
		//		break;
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q16SOLBUF.getreadidx(Q16SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ17(thidx);
		//		break;
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		{
			uint32_t thidx = Q15SOLBUF.getreadidx(Q15SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ16(thidx);
		//		break;
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		// only let warp 0 of each block grab a basesol once per kernel run
		if ((threadIdx.x>>5)==0)
		{
			uint32_t thidx = Q14AUXBUF.getreadidx(Q14AUXCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ15(thidx);
//				break;
				DEBUG_BREAK;
				continue;
			}
		}
#endif

#if 1
		// only let warp 0 of each block grab a basesol once per kernel run
		if ((threadIdx.x>>5)==0)
		{
			uint32_t thidx = Q14SOLBUF.getreadidx(Q14SOLCTL);
			if (thidx != 0xFFFFFFFF)
			{
				stepQ14aux(thidx);
//				break;
				DEBUG_BREAK;
				continue;
			}
		}
#endif

	}
#ifdef USE_PERF_COUNTERS
	while ((clock64()-startclock) < (uint64_t(1)<<34));
#else
	while ((clock64()-startclock) < (uint64_t(1)<<34));
#endif


	backup_controls();
}




void verify_step_computations(int cuda_blocks);
void print_attack_info();


void process_q53_solutions()
{
	using namespace host;
	if (outputfile.empty())
	{
		outputfile = "q61sols.txt";
//		return;
	}

	static size_t q53solcount = 0;
	static size_t oldsize = 0;
	static q61sol_t q61sol;
	static vector<q61sol_t> q61sols;
	static set<q61sol_t> q53uniqsols;

	uint32_t q53idx;
	while ( (q53idx = Q53SOLBUF.getreadidx(Q53SOLCTL)) != 0xFFFFFFFF )
	{
		++q53solcount;
		bool ok = true;
		std::cout << "Q53sol readidx=" << q53idx << std::endl;
		uint32_t m1[80];
		uint32_t Q1[85];
		uint32_t m2[80];
		uint32_t Q2[85];
		// sol: Q49,..,Q53,m37,...,m52

		Q1[QOFF+49] = q53_solutions_buf.get<0>(q53idx);
		Q1[QOFF+50] = q53_solutions_buf.get<1>(q53idx);
		Q1[QOFF+51] = q53_solutions_buf.get<2>(q53idx);
		Q1[QOFF+52] = q53_solutions_buf.get<3>(q53idx);
		Q1[QOFF+53] = q53_solutions_buf.get<4>(q53idx);

		m1[37] = q53_solutions_buf.get<5>(q53idx);
		m1[38] = q53_solutions_buf.get<6>(q53idx);
		m1[39] = q53_solutions_buf.get<7>(q53idx);
		m1[40] = q53_solutions_buf.get<8>(q53idx);
		m1[41] = q53_solutions_buf.get<9>(q53idx);
		m1[42] = q53_solutions_buf.get<10>(q53idx);
		m1[43] = q53_solutions_buf.get<11>(q53idx);
		m1[44] = q53_solutions_buf.get<12>(q53idx);
		m1[45] = q53_solutions_buf.get<13>(q53idx);
		m1[46] = q53_solutions_buf.get<14>(q53idx);
		m1[47] = q53_solutions_buf.get<15>(q53idx);
		m1[48] = q53_solutions_buf.get<16>(q53idx);
		m1[49] = q53_solutions_buf.get<17>(q53idx);
		m1[50] = q53_solutions_buf.get<18>(q53idx);
		m1[51] = q53_solutions_buf.get<19>(q53idx);
		m1[52] = q53_solutions_buf.get<20>(q53idx);

		sha1_me_generalised(m1,37);

		for (int i = 52; i >= 0; --i)
			sha1_step_bw(i, Q1, m1);
		for (int i = -4; i <= 0; ++i)
			if (Q1[QOFF+i] != Qset1mask[QOFF+i])
			{
				ok = false;
				std::cout << "Q53 bad: CV incorrect" << std::endl;
				break;
			}

		ok &= verify(0, 15, 0, Q1, m1, MBR_ORG);

		for (int i = -4; i <= 0; ++i)
			Q2[QOFF+i] = Q1[QOFF+i] + dQ[QOFF+i];
		for (int i = 0; i <= 52; ++i)
		{
			m2[i] = m1[i] ^ DV_DW[i];
			sha1_step(i, Q2, m2);
		}
		for (int i = 49; i <= 53; ++i)
			if (Q1[QOFF+i] != Q2[QOFF+i])
			{
				ok = false;
				for (int j = -4; j <= 53; ++j)
					std::cout << "dQ" << j << "\t:" << std::hex << (Q2[QOFF+j]-Q1[QOFF+j]) << "\t" << std::hex << dQ[QOFF+j] << std::dec << std::endl;
				break;
			}

		if (ok)
		{
			for (unsigned i = 0; i < 16; ++i)
				q61sol.m[i] = m1[i];
			q53uniqsols.insert(q61sol);

#ifndef OUTPUTQ53SOLUTIONS
			// check whether this valid Q53sol is also a Q61sol
			for (int i = 53; i <= 60; ++i)
			{
				sha1_step(i, Q1, m1);
				m2[i] = m1[i] ^ DV_DW[i];
				sha1_step(i, Q2, m2);
			}
			for (int i = 57; i <= 61; ++i)
				if (Q1[QOFF+i] != Q2[QOFF+i])
					ok = false;
#endif

			if (ok)
			{
				q61sols.push_back(q61sol);
			}
		}
	}
	std::cout << "Q53sols=" << q53solcount << " unique=" << q53uniqsols.size() << " Q61sols=" << q61sols.size() << std::endl;

	if (oldsize != q61sols.size())
	{
		oldsize = q61sols.size();
		cout << "Writing " << q61sols.size() << " Q61-solutions to '" << outputfile << "'..." << endl;

		ofstream ofs(outputfile.c_str());
		if (!ofs)
		{
			cout << "Error opening '" << outputfile << ".tmp'!" << endl;
			return;
		}
		for (size_t i = 0; i < q61sols.size(); ++i)
		{
			ofs << encode_q61sol(q61sols[i]) << endl;
		}
	}
}

buffer_q14sol_t  basesol_buf_host;
control_q14sol_t basesol_ctl_host;





bool compiled_with_cuda()
{
	return true;
}

void cuda_main(std::vector<q14sol_t>& q14sols)
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
		cuda_blocks = prop.multiProcessorCount * 2; //(prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
	}
	if (BLOCKS < cuda_blocks)
	{
		cuda_blocks = prop.multiProcessorCount;
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

	cout << "Initializing base solution buffer" << endl;
	size_t basesolcnt = q14sols.size();
	if (basesol_buf_host.size < basesolcnt)
	{
		basesolcnt = basesol_buf_host.size;
	}
	for (size_t i = 0; i < basesolcnt; ++i)
	{
		// m0,...,m15,Q1,...,Q16
		basesol_buf_host.write(basesol_ctl_host, true
			, q14sols[i].m[ 0], q14sols[i].m[ 1], q14sols[i].m[ 2], q14sols[i].m[ 3]
			, q14sols[i].m[ 4], q14sols[i].m[ 5], q14sols[i].m[ 6], q14sols[i].m[ 7]
			, q14sols[i].m[ 8], q14sols[i].m[ 9], q14sols[i].m[10], q14sols[i].m[11]
			, q14sols[i].m[12], q14sols[i].m[13], q14sols[i].m[14], q14sols[i].m[15]
			, q14sols[i].Q[ 0], q14sols[i].Q[ 1], q14sols[i].Q[ 2], q14sols[i].Q[ 3]
			, q14sols[i].Q[ 4], q14sols[i].Q[ 5], q14sols[i].Q[ 6], q14sols[i].Q[ 7]
			, q14sols[i].Q[ 8], q14sols[i].Q[ 9], q14sols[i].Q[10], q14sols[i].Q[11]
			, q14sols[i].Q[12], q14sols[i].Q[13], q14sols[i].Q[14], q14sols[i].Q[15]
			);
	}
	if (basesolcnt == 0)
	{
		cout << "No base solutions.. aborting!" << endl;
		return;
	}
#ifdef USE_MANAGED
	cout << "Moving " << basesolcnt << " base solutions to GPU MANAGED..." << flush;
	// directly copy to variable in HOST memory
	q14_solutions_buf = basesol_buf_host;
	q14_solutions_ctl = basesol_ctl_host;
#else
	cout << "Moving " << basesolcnt << " base solutions to GPU GLOBAL..." << flush;
	// directly copy to variable in GPU GLOBAL memory
	CUDA_ASSERT( cudaMemcpyToSymbol(q14_solutions_buf, &basesol_buf_host, sizeof(basesol_buf_host) ) );
	CUDA_ASSERT( cudaMemcpyToSymbol(q14_solutions_ctl, &basesol_ctl_host, sizeof(basesol_ctl_host) ) );
	CUDA_ASSERT( cudaDeviceSynchronize() );
#endif
	cout << "done." << endl;

	size_t gpumemfree = 0, gpumemtotal = 0;
	CUDA_ASSERT( cudaMemGetInfo ( &gpumemfree, &gpumemtotal ) );
	std::cout << "GPU Memory: " << double(gpumemfree)/double(1<<20) << "MiB of " << double(gpumemtotal)/double(1<<20) << "MiB" << std::endl;

	// use auto for same type deduction, same type is necessary for proper wrap-around behaviour
	uint32_t q14auxoldbufsize[BLOCKS];
	uint32_t q15oldbufsize[BLOCKS];
	uint32_t q16oldbufsize[BLOCKS];
	uint32_t q17oldbufsize[BLOCKS];
	uint32_t q18oldbufsize[BLOCKS];
	uint32_t q19oldbufsize[BLOCKS];
	uint32_t q21oldbufsize[BLOCKS];
	uint32_t q23oldbufsize[BLOCKS];
	uint32_t q26oldbufsize[BLOCKS];
	uint32_t q33oldbufsize[BLOCKS];
	//uint32_t q53oldbufsize[BLOCKS];
	for (unsigned bl = 0; bl < cuda_blocks; ++bl)
	{
		q14auxoldbufsize[bl] = q14aux_solutions_ctl_bu[bl].write_idx;
		q15oldbufsize[bl] = q15_solutions_ctl_bu[bl].write_idx;
		q16oldbufsize[bl] = q16_solutions_ctl_bu[bl].write_idx;
		q17oldbufsize[bl] = q17_solutions_ctl_bu[bl].write_idx;
		q18oldbufsize[bl] = q18_solutions_ctl_bu[bl].write_idx;
		q19oldbufsize[bl] = q19_solutions_ctl_bu[bl].write_idx;
		q21oldbufsize[bl] = q21_solutions_ctl_bu[bl].write_idx;
		q23oldbufsize[bl] = q23_solutions_ctl_bu[bl].write_idx;
		q26oldbufsize[bl] = q26_solutions_ctl_bu[bl].write_idx;
		q33oldbufsize[bl] = q33_solutions_ctl_bu[bl].write_idx;
		//q53oldbufsize[bl] = q53_solutions_ctl_bu[bl].write_idx;
	}
	uint64_t q14auxsols = 0;
	uint64_t q15sols = 0;
	uint64_t q16sols = 0;
	uint64_t q17sols = 0;
	uint64_t q18sols = 0;
	uint64_t q19sols = 0;
	uint64_t q21sols = 0;
	uint64_t q23sols = 0;
	uint64_t q26sols = 0;
	uint64_t q33sols = 0;
	uint64_t q53sols = 0;
//	uint64_t q61sols = 0;

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

		uint64_t basesolsleft = uint32_t(q14_solutions_ctl.write_idx - q14_solutions_ctl.read_idx);
		uint64_t gl_workleft_base = uint32_t(basesolsleft)>>5;
//		uint64_t gl_workleft_q18 = ((q18_solutions_ctl.write_idx - q18_solutions_ctl.read_idx) % q18_solutions_ctl.size) >>5;
//		uint64_t gl_workleft_q19 = ((q19_solutions_ctl.write_idx - q19_solutions_ctl.read_idx) % q19_solutions_ctl.size) >>5;
		uint64_t gl_workleft = gl_workleft_base;// + gl_workleft_q18 + gl_workleft_q19;

//		q18sols += q18_solutions_ctl.write_idx - q18oldbufsize;
//		q19sols += q19_solutions_ctl.write_idx - q19oldbufsize;
//		q18oldbufsize = q18_solutions_ctl.write_idx;
//		q19oldbufsize = q19_solutions_ctl.write_idx;

		uint64_t workleft = gl_workleft;
		for (unsigned bl = 0; bl < cuda_blocks; ++bl)
		{
			workleft += (q14aux_solutions_ctl_bu[bl].write_idx - q14aux_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q15_solutions_ctl_bu[bl].write_idx - q15_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q16_solutions_ctl_bu[bl].write_idx - q16_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q17_solutions_ctl_bu[bl].write_idx - q17_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q18_solutions_ctl_bu[bl].write_idx - q18_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q19_solutions_ctl_bu[bl].write_idx - q19_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q21_solutions_ctl_bu[bl].write_idx - q21_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q23_solutions_ctl_bu[bl].write_idx - q23_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q26_solutions_ctl_bu[bl].write_idx - q26_solutions_ctl_bu[bl].read_idx)>>5;
			workleft += (q33_solutions_ctl_bu[bl].write_idx - q33_solutions_ctl_bu[bl].read_idx)>>5;
			//workleft += (q53_solutions_ctl_bu[bl].write_idx - q53_solutions_ctl_bu[bl].read_idx)>>5;

			q14auxsols += q14aux_solutions_ctl_bu[bl].write_idx - q14auxoldbufsize[bl];
			q15sols += q15_solutions_ctl_bu[bl].write_idx - q15oldbufsize[bl];
			q16sols += q16_solutions_ctl_bu[bl].write_idx - q16oldbufsize[bl];
			q17sols += q17_solutions_ctl_bu[bl].write_idx - q17oldbufsize[bl];
			q18sols += q18_solutions_ctl_bu[bl].write_idx - q18oldbufsize[bl];
			q19sols += q19_solutions_ctl_bu[bl].write_idx - q19oldbufsize[bl];
			q21sols += q21_solutions_ctl_bu[bl].write_idx - q21oldbufsize[bl];
			q23sols += q23_solutions_ctl_bu[bl].write_idx - q23oldbufsize[bl];
			q26sols += q26_solutions_ctl_bu[bl].write_idx - q26oldbufsize[bl];
			q33sols += q33_solutions_ctl_bu[bl].write_idx - q33oldbufsize[bl];
			//q53sols += q53_solutions_ctl_bu[bl].write_idx - q53oldbufsize[bl];

			q14auxoldbufsize[bl] = q14aux_solutions_ctl_bu[bl].write_idx;
			q15oldbufsize[bl] = q15_solutions_ctl_bu[bl].write_idx;
			q16oldbufsize[bl] = q16_solutions_ctl_bu[bl].write_idx;
			q17oldbufsize[bl] = q17_solutions_ctl_bu[bl].write_idx;
			q18oldbufsize[bl] = q18_solutions_ctl_bu[bl].write_idx;
			q19oldbufsize[bl] = q19_solutions_ctl_bu[bl].write_idx;
			q21oldbufsize[bl] = q21_solutions_ctl_bu[bl].write_idx;
			q23oldbufsize[bl] = q23_solutions_ctl_bu[bl].write_idx;
			q26oldbufsize[bl] = q26_solutions_ctl_bu[bl].write_idx;
			q33oldbufsize[bl] = q33_solutions_ctl_bu[bl].write_idx;
			//q53oldbufsize[bl] = q53_solutions_ctl_bu[bl].write_idx;
		}
		q53sols = q53_solutions_ctl.write_idx;
//		q61sols = collision_candidates_ctl.write_idx;
		cout << "Q14a sols:\t" << q14auxsols << "\t" << (double(q14auxsols)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Q15 sols:\t" << q15sols << "\t" << (double(q15sols)/cuda_total_time.time()) << "#/s \t" << log(double(q15sols)/double(q14auxsols))/log(2.0) << endl;
		cout << "Q16 sols:\t" << q16sols << "\t" << (double(q16sols)/cuda_total_time.time()) << "#/s \t" << log(double(q16sols)/double(q15sols))/log(2.0) << endl;
		cout << "Q17 sols:\t" << q17sols << "\t" << (double(q17sols)/cuda_total_time.time()) << "#/s \t" << log(double(q17sols)/double(q16sols))/log(2.0) << endl;
		cout << "Q18 sols:\t" << q18sols << "\t" << (double(q18sols)/cuda_total_time.time()) << "#/s \t" << log(double(q18sols)/double(q17sols))/log(2.0) << endl;
		cout << "Q19 sols:\t" << q19sols << "\t" << (double(q19sols)/cuda_total_time.time()) << "#/s \t" << log(double(q19sols)/double(q18sols))/log(2.0) << endl;
		cout << "Q21 sols:\t" << q21sols << "\t" << (double(q21sols)/cuda_total_time.time()) << "#/s \t" << log(double(q21sols)/double(q19sols))/log(2.0) << endl;
		cout << "Q23 sols:\t" << q23sols << "\t" << (double(q23sols)/cuda_total_time.time()) << "#/s \t" << log(double(q23sols)/double(q21sols))/log(2.0) << endl;
		cout << "Q26 sols:\t" << q26sols << "\t" << (double(q26sols)/cuda_total_time.time()) << "#/s \t" << log(double(q26sols)/double(q23sols))/log(2.0) << endl;
		cout << "Q33 sols:\t" << q33sols << "\t" << (double(q33sols)/cuda_total_time.time()) << "#/s \t" << log(double(q33sols)/double(q26sols))/log(2.0) << endl;
		cout << "Q53 sols:\t" << q53sols << "\t" << (double(q53sols)/cuda_total_time.time()) << "#/s \t" << log(double(q53sols)/double(q33sols))/log(2.0) << endl;
//		cout << "Q61 sols:\t" << q61sols << "\t" << (double(q61sols)/cuda_total_time.time()) << "#/s \t" << log(double(q61sols)/double(q53sols))/log(2.0) << endl;

		process_q53_solutions();

#ifdef USE_PERF_COUNTERS
		show_performance_counters();
#endif

		// exit if base solutions have been exhausted
		// !! NOTE THAT THERE MAY STILL BE SOME OTHER WORK LEFT !!

		cout << "Basesolutions left: " << basesolsleft << "\t" << (double(q14_solutions_ctl.read_idx)/cuda_total_time.time()) << "#/s" << endl;
		cout << "Global work left: B:" << gl_workleft_base /*<< "  Q18:" << gl_workleft_q18 << "  Q19:" << gl_workleft_q19*/ << endl;

		if (gl_workleft == 0)
		{
			cout << "Exhausted work!" << endl;
			break;
		}

//		boost::this_thread::sleep_for( boost::chrono::seconds(1) );
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




bool verify_Q15(int block, size_t read_idx)
{
	using namespace host;
	uint32_t m[80];
	uint32_t Q[85];

	size_t q14idx = q15_solutions_buf[block].get<11>(read_idx);

	m[0] = q14_solutions_buf.get<0>(q14idx);
	m[1] = q14_solutions_buf.get<1>(q14idx);
	m[2] = q14_solutions_buf.get<2>(q14idx);
	m[3] = q14_solutions_buf.get<3>(q14idx);
	m[4] = q14_solutions_buf.get<4>(q14idx);
	m[5] = q14_solutions_buf.get<5>(q14idx);
	m[6] = q14_solutions_buf.get<6>(q14idx);
	m[7] = q14_solutions_buf.get<7>(q14idx);
	m[8] = q14_solutions_buf.get<8>(q14idx);
	m[9] = q14_solutions_buf.get<9>(q14idx);

	m[10] = q15_solutions_buf[block].get<5>(read_idx);
	m[11] = q15_solutions_buf[block].get<6>(read_idx);
	m[12] = q15_solutions_buf[block].get<7>(read_idx);
	m[13] = q15_solutions_buf[block].get<8>(read_idx);
	m[14] = q15_solutions_buf[block].get<9>(read_idx);
	m[15] = q15_solutions_buf[block].get<10>(read_idx);

	for (int i = -4; i <= 0; ++i)
		Q[QOFF+i] = Qset1mask[QOFF+i];

	Q[QOFF+1] = q14_solutions_buf.get<15+1>(q14idx);
	Q[QOFF+2] = q14_solutions_buf.get<15+2>(q14idx);
	Q[QOFF+3] = q14_solutions_buf.get<15+3>(q14idx);
	Q[QOFF+4] = q14_solutions_buf.get<15+4>(q14idx);
	Q[QOFF+5] = q14_solutions_buf.get<15+5>(q14idx);
	Q[QOFF+6] = q14_solutions_buf.get<15+6>(q14idx);
	Q[QOFF+7] = q14_solutions_buf.get<15+7>(q14idx);
	Q[QOFF+8] = q14_solutions_buf.get<15+8>(q14idx);
	Q[QOFF+9] = q14_solutions_buf.get<15+9>(q14idx);
	Q[QOFF+10] = q14_solutions_buf.get<15+10>(q14idx);

	Q[QOFF+11] = q15_solutions_buf[block].get<0>(read_idx);
	Q[QOFF+12] = q15_solutions_buf[block].get<1>(read_idx);
	Q[QOFF+13] = q15_solutions_buf[block].get<2>(read_idx);
	Q[QOFF+14] = q15_solutions_buf[block].get<3>(read_idx);
	Q[QOFF+15] = q15_solutions_buf[block].get<4>(read_idx);

	sha1_step<15>(Q,m);

	return verify(0, 15, 15, Q, m, MBR_Q17NB);
}

bool verify_Q33(int block, size_t read_idx)
{
	using namespace host;
	uint32_t m1[80];
	uint32_t Q1[85];
	uint32_t m2[80];
	uint32_t Q2[85];

	// sol: Q29,..,Q33,m17,...,m32

	Q1[QOFF+29] = q33_solutions_buf[block].get<0>(read_idx);
	Q1[QOFF+30] = q33_solutions_buf[block].get<1>(read_idx);
	Q1[QOFF+31] = q33_solutions_buf[block].get<2>(read_idx);
	Q1[QOFF+32] = q33_solutions_buf[block].get<3>(read_idx);
	Q1[QOFF+33] = q33_solutions_buf[block].get<4>(read_idx);

	m1[17] = q33_solutions_buf[block].get<5>(read_idx);
	m1[18] = q33_solutions_buf[block].get<6>(read_idx);
	m1[19] = q33_solutions_buf[block].get<7>(read_idx);
	m1[20] = q33_solutions_buf[block].get<8>(read_idx);
	m1[21] = q33_solutions_buf[block].get<9>(read_idx);
	m1[22] = q33_solutions_buf[block].get<10>(read_idx);
	m1[23] = q33_solutions_buf[block].get<11>(read_idx);
	m1[24] = q33_solutions_buf[block].get<12>(read_idx);
	m1[25] = q33_solutions_buf[block].get<13>(read_idx);
	m1[26] = q33_solutions_buf[block].get<14>(read_idx);
	m1[27] = q33_solutions_buf[block].get<15>(read_idx);
	m1[28] = q33_solutions_buf[block].get<16>(read_idx);
	m1[29] = q33_solutions_buf[block].get<17>(read_idx);
	m1[30] = q33_solutions_buf[block].get<18>(read_idx);
	m1[31] = q33_solutions_buf[block].get<19>(read_idx);
	m1[32] = q33_solutions_buf[block].get<20>(read_idx);

	sha1_me_generalised(m1,17);

	for (int i = 32; i >= 0; --i)
		sha1_step_bw(i, Q1, m1);
	for (int i = -4; i <= 0; ++i)
		if (Q1[QOFF+i] != Qset1mask[QOFF+i])
			return false;

	//verify(0, 23, 23, Q1, m1, MBR_ORG);

	for (int i = -4; i <= 0; ++i)
		Q2[QOFF+i] = Q1[QOFF+i] + dQ[QOFF+i];
	for (int i = 0; i <= 32; ++i)
	{
		m2[i] = m1[i] ^ DV_DW[i];
		sha1_step(i, Q2, m2);
	}
	for (int i = 29; i <= 33; ++i)
		if (Q1[QOFF+i] != Q2[QOFF+i])
		{
			for (int j = -4; j <= 33; ++j)
				std::cout << "dQ" << j << "\t:" << std::hex << (Q2[QOFF+j]-Q1[QOFF+j]) << "\t" << std::hex << dQ[QOFF+j] << std::dec << std::endl;

			return false;
		}

	return true;
}























void verify_step_computations(int cuda_blocks)
{
	for (unsigned block = 0; block < cuda_blocks; ++block)
	{
		cout << "======== Verifying block " << block << endl;

		cout << "Base solutions left: " << (q14_solutions_ctl.write_idx - q14_solutions_ctl.read_idx) << setfill(' ') << endl;
		size_t q15checked = 0, q15ok = 0;
		uint32_t q15count = q15_solutions_ctl_bu[block].write_idx - q15_solutions_ctl_bu[block].read_idx;
		//cout << q15_solutions_ctl_bu[block].read_idx << " " << q15_solutions_ctl_bu[block].write_idx << " " << q15count << endl;
		for (uint32_t i = q15_solutions_ctl_bu[block].read_idx; i != q15_solutions_ctl_bu[block].write_idx; ++i)
		{
			if (verify_Q15(block, i))
				++q15ok;
			++q15checked;
			if (i - q15_solutions_ctl_bu[block].read_idx > q15_solutions_ctl_bu[block].size)
			break;
		}
		cout << "Verified " << setw(10) << q15checked << " out of " << setw(10) << q15count << " Q18 solutions: " << q15ok << " OK" << endl;

		size_t q33checked = 0, q33ok = 0;
		uint32_t q33count = q33_solutions_ctl_bu[block].write_idx - q33_solutions_ctl_bu[block].read_idx;
		//cout << q15_solutions_ctl_bu[block].read_idx << " " << q15_solutions_ctl_bu[block].write_idx << " " << q15count << endl;
		for (uint32_t i = q33_solutions_ctl_bu[block].read_idx; i != q33_solutions_ctl_bu[block].write_idx; ++i)
		{
			if (verify_Q33(block, i))
				++q33ok;
			++q33checked;
			if (i - q33_solutions_ctl_bu[block].read_idx > q33_solutions_ctl_bu[block].size)
			break;
		}
		cout << "Verified " << setw(10) << q33checked << " out of " << setw(10) << q33count << " Q33 solutions: " << q33ok << " OK" << endl;

	}
}

#endif // VERIFY_GPU_RESULTS
