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
