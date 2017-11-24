/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
            (C) 2016 Pierre Karpman

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

#ifndef MAIN_HPP
#define MAIN_HPP

#define MINQ14SOLPERJOB (1<<10)
#define Q14BUFSIZE      (1<<16)

//#define SILENTERROR

// enable for unit testing
//#define OUTPUTQ53SOLUTIONS

#include "types.hpp"
#include "sha1detail.hpp"

#include <base64.hpp>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

void gpusha1benchmark();

struct q13sol_t {
	uint32_t m[16]; // m0,...,m15
};

struct q14sol_t {
	uint32_t m[16]; // m0,...,m15
	uint32_t Q[16]; // Q1,...,Q16
};

struct q33sol_t {
	uint32_t Q[5]; // Q29,...,Q33
	uint32_t m[16]; // m17,...,m32
};

struct q53sol_t {
	uint32_t Q[5]; // Q49,...,Q53
	uint32_t m[16]; // m37,...,m52
};

struct q61sol_t {
//	uint32_t Q[5]; // Q57,...,Q61
	uint32_t m[16]; // m0,...,m15
	bool operator<(const q61sol_t& r) const
	{
		for (unsigned i = 0; i < 16; ++i)
			if (m[i] != r.m[i])
				return m[i] < r.m[i];
		return false;
	}
};

extern int cuda_device, cuda_blocks, cuda_threads_per_block, cuda_scheduler;

extern std::vector<std::string> inputfile;
extern std::string outputfile;

bool compiled_with_cuda();

void cuda_query();

void cuda_main(std::vector<q14sol_t>& q14sols);





// find q13solutions in basesolgen.cpp
// calls found_q13sol()
// returns never
void gen_q13sols();

// called by basesolgen.cpp whenever a q13-solution is found
// for further processing
extern int max_basesols;
extern std::vector<q13sol_t> q13sols;
void process_q13sol(const uint32_t m1[80], const uint32_t Q1[85]);

// call step13nb to exploit step 13 neutralbits to satisfy Q14
// calls process_q14sol for each Q14-solution
void step13nb(const uint32_t m1[80], const uint32_t Q1[85]);
extern std::vector<q14sol_t> q14sols;
void process_q14sol(const uint32_t m1[80], const uint32_t Q1[85]);

// verifies whether q13sol is OK (satisfies stateconds and msgbitrel for mainblockoffset 1)
// and neutral bits are set to 0
// and has backward error probability under neutral bits < 1%
enum mbrset_t { MBR_ORG, MBR_BOOM, MBR_Q17NB, MBR_Q14NB };
bool verify(int firststep, int laststep, int lastQ, const uint32_t* mQ1, const uint32_t* mm1, mbrset_t mbrset = MBR_Q14NB );

inline bool verify(const q13sol_t& q13sol)
{
	uint32_t Q[21] = { /*0*/ 0xce2969ef, /*1*/ 0x7b1facd1, /*2*/ 0xaf216457, /*3*/ 0xffed5352, /*4*/ 0x8d64d617 }; // taken from Qset1cond
	uint32_t m[16];
	memcpy(m, q13sol.m, 16*4);
	for (unsigned t = 0; t < 16; ++t)
		hc::sha1_step_round1(t,Q,m);
	return verify(0, 15, 13, Q, m, MBR_Q14NB);
}
inline bool verify(const q14sol_t& q14sol)
{
	uint32_t Q[21] = { /*0*/ 0xce2969ef, /*1*/ 0x7b1facd1, /*2*/ 0xaf216457, /*3*/ 0xffed5352, /*4*/ 0x8d64d617 }; // taken from Qset1cond
	uint32_t m[16];
	memcpy(m, q14sol.m, 16*4);
	for (unsigned t = 0; t < 16; ++t)
		hc::sha1_step_round1(t,Q,m);
	for (int t = 1; t <= 16; ++t)
		if (Q[4+t] != q14sol.Q[t-1])
			std::cerr << "WHAT?!?!?" << std::endl;
	return verify(0, 15, 14, Q, m, MBR_Q17NB);
}
inline void step13nb(const q13sol_t& q13sol)
{
	uint32_t m1[80];
	uint32_t Q1[85] = { /*0*/ 0xce2969ef, /*1*/ 0x7b1facd1, /*2*/ 0xaf216457, /*3*/ 0xffed5352, /*4*/ 0x8d64d617 }; // taken from Qset1cond
	memcpy(m1, q13sol.m, 16*4);
	for (unsigned t = 0; t < 16; ++t)
		hc::sha1_step_round1(t,Q1,m1);
	verify(0, 15, 13, Q1, m1, MBR_Q14NB);
	step13nb(m1, Q1);
}

static inline std::string encode_q13sol(const q13sol_t& bs)
{
	return "B!" + base64_encode(std::string((const char*)(&bs), sizeof(q13sol_t)));
}
static inline q13sol_t decode_q13sol(const std::string& in)
{
	if (in.substr(0, 2) != "B!")
		throw std::runtime_error("decode_q13sol(): input string doesn't have required marker 'B!'");
	std::string dec = base64_decode(in.substr(2));
	q13sol_t out;
	if (dec.size() != sizeof(q13sol_t))
		throw std::runtime_error("decode_q13sol(): decoded binary string has incorrect length!");
	memcpy(&out, dec.c_str(), sizeof(q13sol_t));
	return out;
}

static inline std::string encode_q61sol(const q61sol_t& s)
{
	return "Q!" + base64_encode(std::string((const char*)(&s), sizeof(q61sol_t)));
}
static inline q61sol_t decode_q61sol(const std::string& in)
{
	if (in.substr(0, 2) != "Q!")
		throw std::runtime_error("decode_q61sol(): input string doesn't have required marker 'Q!'");
	std::string dec = base64_decode(in.substr(2));
	q61sol_t out;
	if (dec.size() != sizeof(q61sol_t))
		throw std::runtime_error("decode_q61sol(): decoded binary string has incorrect length!");
	memcpy(&out, dec.c_str(), sizeof(q61sol_t));
	return out;
}

#endif // MAIN_HPP
