/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

#ifndef HASHCLASH_TYPES_HPP
#define HASHCLASH_TYPES_HPP

#ifdef __CUDACC__
#pragma message("CUDA compiler detected.")
#define NOSERIALIZATION
#define FUNC_PREFIX __device__ __host__
#include <stdint.h>
#else
#define FUNC_PREFIX
#include <stdint.h>
#endif // __CUDACC__

#ifdef WIN32
#include <intrin.h> 
inline unsigned __builtin_popcount(uint32_t w) { return __popcnt(w); }
#endif

namespace hc {

	FUNC_PREFIX inline uint32_t rotate_right(const uint32_t x, const unsigned n)
	{ return (x>>n) | (x<<(32-n)); }
	FUNC_PREFIX inline uint32_t rotate_left(const uint32_t x, const unsigned n)
	{ return (x<<n) | (x>>(32-n)); }
	FUNC_PREFIX inline uint64_t rotate_right(const uint64_t x, const unsigned n)
	{ return (x>>n) | (x<<(64-n)); }
	FUNC_PREFIX inline uint64_t rotate_left(const uint64_t x, const unsigned n)
	{ return (x<<n) | (x>>(64-n)); }

	inline unsigned hw(uint32_t w) { return __builtin_popcount(w); }

} // namespace hc


#endif // HASHCLASH_TYPES_HPP
