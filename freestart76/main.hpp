/*****
  Copyright (C) 2015 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
            (C) 2015 Pierre Karpman, INRIA France/Nanyang Technological University Singapore (-2016), CWI (2016/2017), L'Universite Grenoble Alpes (2017-)

  This file is part of sha1_gpu_nearcollisionattacks source-code and released under the MIT License
*****/

#ifndef MAIN_HPP
#define MAIN_HPP

#include "types.hpp"

#include <vector>
#include <string>

void gpusha1benchmark();

struct basesol_t {
	uint32_t Q[6];  // Q12,..,Q17
	uint32_t m[16]; // W6,...,W21
};

struct q56sol_t {
	uint32_t Q[5]; // Q32,...,Q36
	uint32_t m[16]; // m20,...,m35
};

extern int cuda_device, cuda_blocks, cuda_threads_per_block;
extern std::vector<std::string> inputfile;
extern std::string outputfile;
extern bool disable_backwards_filter;
extern int cuda_scheduler;
extern int max_basesols;

bool compiled_with_cuda();

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
void found_basesol(uint32_t main_m1[80], uint32_t main_Q1[85], unsigned mainblockoffset);

// verifies whether basesol is OK (satisfies stateconds and msgbitrel for mainblockoffset 1)
// and neutral bits are set to 0
// and has backward error probability under neutral bits < 1%
extern bool BASESOL_OK;
bool verify(basesol_t basesol);

#endif // MAIN_HPP
