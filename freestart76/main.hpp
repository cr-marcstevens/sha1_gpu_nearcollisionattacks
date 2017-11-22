#ifndef MAIN_HPP
#define MAIN_HPP


#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>

#include <thread>

#include "types.hpp"
#include "rng.hpp"
#include "timer.hpp"
#include "sha1detail.hpp"

using namespace hc;
using namespace std;

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
void found_basesol(uint32_t main_m1[80], uint32_t main_Q1[85], unsigned mainblockoffset);

// verifies whether basesol is OK (satisfies stateconds and msgbitrel for mainblockoffset 1)
// and neutral bits are set to 0
// and has backward error probability under neutral bits < 1%
extern bool BASESOL_OK;
bool verify(basesol_t basesol);

#endif // MAIN_HPP
