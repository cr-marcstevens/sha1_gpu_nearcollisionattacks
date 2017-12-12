/*****
  Copyright (C) 2015 Pierre Karpman, INRIA France/Nanyang Technological University Singapore (-2016), CWI (2016/2017), L'Universite Grenoble Alpes (2017-)
            (C) 2015 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1_gpu_nearcollisionattacks source-code and released under the MIT License
*****/

#include <iostream>
#include <stdint.h>
#include <vector>

struct basesol_t {
uint32_t Q[6];
uint32_t m[16];
};

struct q14sol_t {
uint32_t Q[16]; 
uint32_t m[16];
};

bool compiled_with_cuda()
{
	return false;
}

void cuda_query()
{
	std::cerr << "cuda_query(): not compiled in non-CUDA basesolution generator program" << std::endl;
}

void cuda_main(std::vector<basesol_t>& basesols)
{
	std::cerr << "cuda_main(): not compiled for non-CUDA basesolution generator program" << std::endl;
}

void cuda_main(std::vector<q14sol_t>& q14sols)
{
	std::cerr << "cuda_main(): not compiled for non-CUDA basesolution generator program" << std::endl;
}

void gpusha1benchmark()
{
	std::cerr << "gpusha1benchmark(): not compiled for non-CUDA basesolution generator program" << std::endl;
}
