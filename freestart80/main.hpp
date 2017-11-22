/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
            (C) 2016 Pierre Karpman

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

#ifndef MAIN_HPP
#define MAIN_HPP


#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
#include <cstring>

#include "types.hpp"
#include "rng.hpp"
#include "timer.hpp"
#include "sha1detail.hpp"
#include "base64.hpp"

using namespace hc;
using namespace std;

void gpusha1benchmark();

struct basesol_t {
	uint32_t Q[6];  // Q12,..,Q17
	uint32_t m[16]; // W5,...,W20
};

struct q40sol_t {
	uint32_t Q[5]; // Q36,...,Q40
	uint32_t m[16]; // m24,...,m39
};

struct q60sol_t {
	uint32_t Q[5]; // Q56,...,Q60
	uint32_t m[16]; // m44,...,m59
};

extern int cuda_device, cuda_blocks, cuda_threads_per_block;
extern vector<string> inputfile;
extern string outputfile;
extern bool disable_backwards_filter;
extern int cuda_scheduler;

void cuda_query();

#define BASESOLCOUNT (1<<20)
void cuda_main(std::vector<basesol_t>& basesols);
void cpu_main(std::vector<basesol_t>& basesols);

// find basesolutions in find_basesolutions_mbo1.cpp
// calls found_basesol()
// returns never
void start_attack();

// called by find_basesolution_mbo1.cpp whenever a base solution is found
// for further processing
extern int max_basesols;
void found_basesol(uint32_t main_m1[80], uint32_t main_Q1[85], unsigned mainblockoffset);

// verifies whether basesol is OK (satisfies stateconds and msgbitrel for mainblockoffset 1)
// and neutral bits are set to 0
// and has backward error probability under neutral bits < 1%
extern bool BASESOL_OK;
bool verify(basesol_t basesol);


static inline std::string encode_basesol(const basesol_t& bs)
{
	return "B!" + base64_encode(std::string((const char*)(&bs), sizeof(basesol_t)));
}
static inline basesol_t decode_basesol(const std::string& in)
{
	if (in.substr(0, 2) != "B!")
		throw std::runtime_error("decode_basesol(): input string doesn't have required marker 'B!'");
	std::string dec = base64_decode(in.substr(2));
	basesol_t out;
	if (dec.size() != sizeof(basesol_t))
		throw std::runtime_error("decode_basesol(): decoded binary string has incorrect length!");
	memcpy(&out, dec.c_str(), sizeof(basesol_t));
	return out;
}

static inline std::string encode_q60sol(const q60sol_t& s)
{
	return "Q!" + base64_encode(std::string((const char*)(&s), sizeof(q60sol_t)));
}
static inline q60sol_t decode_q60sol(const std::string& in)
{
	if (in.substr(0, 2) != "Q!")
		throw std::runtime_error("decode_basesol(): input string doesn't have required marker 'B!'");
	std::string dec = base64_decode(in.substr(2));
	q60sol_t out;
	if (dec.size() != sizeof(q60sol_t))
		throw std::runtime_error("decode_q60sol(): decoded binary string has incorrect length!");
	memcpy(&out, dec.c_str(), sizeof(q60sol_t));
	return out;
}

#endif // MAIN_HPP
