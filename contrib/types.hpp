/**************************************************************************\
|
|    Copyright (C) 2009 Marc Stevens
|    https://github.com/cr-marcstevens/hashclash
|
|    This program is free software: you can redistribute it and/or modify
|    it under the terms of the GNU General Public License as published by
|    the Free Software Foundation, either version 3 of the License, or
|    (at your option) any later version.
|
|    This program is distributed in the hope that it will be useful,
|    but WITHOUT ANY WARRANTY; without even the implied warranty of
|    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
|    GNU General Public License for more details.
|
|    You should have received a copy of the GNU General Public License
|    along with this program.  If not, see <http://www.gnu.org/licenses/>.
|
\**************************************************************************/

#ifndef HASHCLASH_TYPES_HPP
#define HASHCLASH_TYPES_HPP

#ifdef __CUDACC__
#pragma message("CUDA compiler detected.")
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

namespace hashclash {

	FUNC_PREFIX inline uint32_t rotate_right(const uint32_t x, const unsigned n)
	{ return (x>>n) | (x<<(32-n)); }
	FUNC_PREFIX inline uint32_t rotate_left(const uint32_t x, const unsigned n)
	{ return (x<<n) | (x>>(32-n)); }
	FUNC_PREFIX inline uint64_t rotate_right(const uint64_t x, const unsigned n)
	{ return (x>>n) | (x<<(64-n)); }
	FUNC_PREFIX inline uint64_t rotate_left(const uint64_t x, const unsigned n)
	{ return (x<<n) | (x>>(64-n)); }

        inline unsigned hw(uint32_t w) { return __builtin_popcount(w); }

} // namespace hashclash

#endif // HASHCLASH_TYPES_HPP
