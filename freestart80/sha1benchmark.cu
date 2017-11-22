/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
            (C) 2016 Pierre Karpman

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

#include <iomanip>

#include "main.hpp"

#define MAXBLOCKS 52
#define MAXTHREADSPERBLOCK 1024
#define MAXGPUTHREADS (MAXBLOCKS*MAXTHREADSPERBLOCK)

#define CUDA_ASSERT(s) 	{ cudaError_t err = s; if (err != cudaSuccess) { throw std::runtime_error("CUDA command returned: " + string(cudaGetErrorString(err)) + "!"); }  }


__device__ __constant__	const uint32_t sha1_iv2[] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };

__device__ __constant__ const uint32_t me[80] = { 0,0,0,0 ,0,0,0,0 ,0,0,0,0 ,0,0,0,0 };
__device__ __managed__ uint32_t ihv[5][MAXGPUTHREADS];

#define HASHCLASH_SHA1COMPRESS_ROUND1_STEP(a, b, c, d, e, m, t) \
    e += rotate_left(a, 5) + sha1_f1(b,c,d) + 0x5A827999 + m[t]; b = rotate_left(b, 30);
#define HASHCLASH_SHA1COMPRESS_ROUND2_STEP(a, b, c, d, e, m, t) \
    e += rotate_left(a, 5) + sha1_f2(b,c,d) + 0x6ED9EBA1 + m[t]; b = rotate_left(b, 30);
#define HASHCLASH_SHA1COMPRESS_ROUND3_STEP(a, b, c, d, e, m, t) \
    e += rotate_left(a, 5) + sha1_f3(b,c,d) + 0x8F1BBCDC + m[t]; b = rotate_left(b, 30);
#define HASHCLASH_SHA1COMPRESS_ROUND4_STEP(a, b, c, d, e, m, t) \
    e += rotate_left(a, 5) + sha1_f4(b,c,d) + 0xCA62C1D6 + m[t]; b = rotate_left(b, 30);


#define HASHCLASH_SHA1COMPRESS_ROUND1_STEPB(a, b, c, d, e, m) \
    e += rotate_left(a, 5) + sha1_f1(b,c,d) + 0x5A827999 + m; b = rotate_left(b, 30);
#define HASHCLASH_SHA1COMPRESS_ROUND2_STEPB(a, b, c, d, e, m) \
    e += rotate_left(a, 5) + sha1_f2(b,c,d) + 0x6ED9EBA1 + m; b = rotate_left(b, 30);
#define HASHCLASH_SHA1COMPRESS_ROUND3_STEPB(a, b, c, d, e, m) \
    e += rotate_left(a, 5) + sha1_f3(b,c,d) + 0x8F1BBCDC + m; b = rotate_left(b, 30);
#define HASHCLASH_SHA1COMPRESS_ROUND4_STEPB(a, b, c, d, e, m) \
    e += rotate_left(a, 5) + sha1_f4(b,c,d) + 0xCA62C1D6 + m; b = rotate_left(b, 30);

__device__ inline uint32_t sha1_mess(uint32_t m_3, uint32_t m_8, uint32_t m_14, uint32_t m_16)
{
        return rotate_left(m_3 ^ m_8 ^ m_14 ^ m_16, 1);
}


__global__ void sha1compress_test(uint32_t count)
	{
		const uint32_t threadid = threadIdx.x + blockIdx.x*blockDim.x;

		for (unsigned i = 0; i < 5; ++i)
			ihv[i][threadid] = sha1_iv2[i];

		for (uint32_t i = 0; i < count; ++i)
		{
		uint32_t a = ihv[0][threadid]; uint32_t b = ihv[1][threadid]; uint32_t c = ihv[2][threadid]; uint32_t d = ihv[3][threadid]; uint32_t e = ihv[4][threadid];

		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( a, b, c, d, e, me,  0 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( e, a, b, c, d, me,  1 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( d, e, a, b, c, me,  2 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( c, d, e, a, b, me,  3 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( b, c, d, e, a, me,  4 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( a, b, c, d, e, me,  5 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( e, a, b, c, d, me,  6 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( d, e, a, b, c, me,  7 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( c, d, e, a, b, me,  8 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( b, c, d, e, a, me,  9 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( a, b, c, d, e, me, 10 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( e, a, b, c, d, me, 11 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( d, e, a, b, c, me, 12 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( c, d, e, a, b, me, 13 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( b, c, d, e, a, me, 14 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( a, b, c, d, e, me, 15 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( e, a, b, c, d, me, 16 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( d, e, a, b, c, me, 17 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( c, d, e, a, b, me, 18 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( b, c, d, e, a, me, 19 );

		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( a, b, c, d, e, me, 20 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( e, a, b, c, d, me, 21 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( d, e, a, b, c, me, 22 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( c, d, e, a, b, me, 23 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( b, c, d, e, a, me, 24 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( a, b, c, d, e, me, 25 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( e, a, b, c, d, me, 26 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( d, e, a, b, c, me, 27 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( c, d, e, a, b, me, 28 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( b, c, d, e, a, me, 29 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( a, b, c, d, e, me, 30 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( e, a, b, c, d, me, 31 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( d, e, a, b, c, me, 32 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( c, d, e, a, b, me, 33 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( b, c, d, e, a, me, 34 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( a, b, c, d, e, me, 35 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( e, a, b, c, d, me, 36 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( d, e, a, b, c, me, 37 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( c, d, e, a, b, me, 38 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( b, c, d, e, a, me, 39 );

		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( a, b, c, d, e, me, 40 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( e, a, b, c, d, me, 41 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( d, e, a, b, c, me, 42 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( c, d, e, a, b, me, 43 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( b, c, d, e, a, me, 44 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( a, b, c, d, e, me, 45 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( e, a, b, c, d, me, 46 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( d, e, a, b, c, me, 47 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( c, d, e, a, b, me, 48 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( b, c, d, e, a, me, 49 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( a, b, c, d, e, me, 50 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( e, a, b, c, d, me, 51 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( d, e, a, b, c, me, 52 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( c, d, e, a, b, me, 53 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( b, c, d, e, a, me, 54 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( a, b, c, d, e, me, 55 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( e, a, b, c, d, me, 56 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( d, e, a, b, c, me, 57 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( c, d, e, a, b, me, 58 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( b, c, d, e, a, me, 59 );

		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( a, b, c, d, e, me, 60 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( e, a, b, c, d, me, 61 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( d, e, a, b, c, me, 62 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( c, d, e, a, b, me, 63 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( b, c, d, e, a, me, 64 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( a, b, c, d, e, me, 65 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( e, a, b, c, d, me, 66 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( d, e, a, b, c, me, 67 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( c, d, e, a, b, me, 68 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( b, c, d, e, a, me, 69 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( a, b, c, d, e, me, 70 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( e, a, b, c, d, me, 71 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( d, e, a, b, c, me, 72 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( c, d, e, a, b, me, 73 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( b, c, d, e, a, me, 74 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( a, b, c, d, e, me, 75 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( e, a, b, c, d, me, 76 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( d, e, a, b, c, me, 77 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( c, d, e, a, b, me, 78 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( b, c, d, e, a, me, 79 );

		ihv[0][threadid] += a; ihv[1][threadid] += b; ihv[2][threadid] += c; ihv[3][threadid] += d; ihv[4][threadid] += e;
		}
	}




__global__ void sha1compress_with_me_test(uint32_t count)
	{
		const uint32_t threadid = threadIdx.x + blockIdx.x*blockDim.x;

		for (unsigned i = 0; i < 5; ++i)
			ihv[i][threadid] = sha1_iv2[i];

		for (uint32_t i = 0; i < count; ++i)
		{
		uint32_t a = ihv[0][threadid]; uint32_t b = ihv[1][threadid]; uint32_t c = ihv[2][threadid]; uint32_t d = ihv[3][threadid]; uint32_t e = ihv[4][threadid];
		uint32_t m0 = me[0];
		uint32_t m1 = me[1];
		uint32_t m2 = me[2];
		uint32_t m3 = me[3];
		uint32_t m4 = me[4];
		uint32_t m5 = me[5];
		uint32_t m6 = me[6];
		uint32_t m7 = me[7];
		uint32_t m8 = me[8];
		uint32_t m9 = me[9];
		uint32_t m10 = me[10];
		uint32_t m11 = me[11];
		uint32_t m12 = me[12];
		uint32_t m13 = me[13];
		uint32_t m14 = me[14];
		uint32_t m15 = me[15];

		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( a, b, c, d, e, m0 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( e, a, b, c, d, m1 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( d, e, a, b, c, m2 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( c, d, e, a, b, m3 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( b, c, d, e, a, m4 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( a, b, c, d, e, m5 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( e, a, b, c, d, m6 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( d, e, a, b, c, m7 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( c, d, e, a, b, m8 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( b, c, d, e, a, m9 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( a, b, c, d, e, m10 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( e, a, b, c, d, m11 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( d, e, a, b, c, m12 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( c, d, e, a, b, m13 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( b, c, d, e, a, m14 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( a, b, c, d, e, m15 );

{
        m0 = sha1_mess(m13, m8, m2, m0);
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( e, a, b, c, d, m0 );
        m1 = sha1_mess(m14, m9, m3, m1);
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( d, e, a, b, c, m1 );
        m2 = sha1_mess(m15, m10, m4, m2);
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( c, d, e, a, b, m2 );
        m3 = sha1_mess(m0, m11, m5, m3);
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( b, c, d, e, a, m3 );
        m4 = sha1_mess(m1, m12, m6, m4);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( a, b, c, d, e, m4 );
        m5 = sha1_mess(m2, m13, m7, m5);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( e, a, b, c, d, m5 );
        m6 = sha1_mess(m3, m14, m8, m6);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( d, e, a, b, c, m6 );
        m7 = sha1_mess(m4, m15, m9, m7);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( c, d, e, a, b, m7 );
        m8 = sha1_mess(m5, m0, m10, m8);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( b, c, d, e, a, m8 );
        m9 = sha1_mess(m6, m1, m11, m9);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( a, b, c, d, e, m9 );
        m10 = sha1_mess(m7, m2, m12, m10);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( e, a, b, c, d, m10 );
        m11 = sha1_mess(m8, m3, m13, m11);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( d, e, a, b, c, m11 );
        m12 = sha1_mess(m9, m4, m14, m12);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( c, d, e, a, b, m12 );
        m13 = sha1_mess(m10, m5, m15, m13);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( b, c, d, e, a, m13 );
        m14 = sha1_mess(m11, m6, m0, m14);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( a, b, c, d, e, m14 );
        m15 = sha1_mess(m12, m7, m1, m15);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( e, a, b, c, d, m15 );
}
{
        m0 = sha1_mess(m13, m8, m2, m0);
        m1 = sha1_mess(m14, m9, m3, m1);
        m2 = sha1_mess(m15, m10, m4, m2);
        m3 = sha1_mess(m0, m11, m5, m3);
        m4 = sha1_mess(m1, m12, m6, m4);
        m5 = sha1_mess(m2, m13, m7, m5);
        m6 = sha1_mess(m3, m14, m8, m6);
        m7 = sha1_mess(m4, m15, m9, m7);
        m8 = sha1_mess(m5, m0, m10, m8);
        m9 = sha1_mess(m6, m1, m11, m9);
        m10 = sha1_mess(m7, m2, m12, m10);
        m11 = sha1_mess(m8, m3, m13, m11);
        m12 = sha1_mess(m9, m4, m14, m12);
        m13 = sha1_mess(m10, m5, m15, m13);
        m14 = sha1_mess(m11, m6, m0, m14);
        m15 = sha1_mess(m12, m7, m1, m15);

		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( d, e, a, b, c, m0 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( c, d, e, a, b, m1 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( b, c, d, e, a, m2 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( a, b, c, d, e, m3  );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( e, a, b, c, d, m4  );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( d, e, a, b, c, m5  );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( c, d, e, a, b, m6  );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( b, c, d, e, a, m7  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( a, b, c, d, e, m8  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( e, a, b, c, d, m9  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( d, e, a, b, c, m10 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( c, d, e, a, b, m11 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( b, c, d, e, a, m12 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( a, b, c, d, e, m13 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( e, a, b, c, d, m14 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( d, e, a, b, c, m15 );
}
{
        m0 = sha1_mess(m13, m8, m2, m0);
        m1 = sha1_mess(m14, m9, m3, m1);
        m2 = sha1_mess(m15, m10, m4, m2);
        m3 = sha1_mess(m0, m11, m5, m3);
        m4 = sha1_mess(m1, m12, m6, m4);
        m5 = sha1_mess(m2, m13, m7, m5);
        m6 = sha1_mess(m3, m14, m8, m6);
        m7 = sha1_mess(m4, m15, m9, m7);
        m8 = sha1_mess(m5, m0, m10, m8);
        m9 = sha1_mess(m6, m1, m11, m9);
        m10 = sha1_mess(m7, m2, m12, m10);
        m11 = sha1_mess(m8, m3, m13, m11);
        m12 = sha1_mess(m9, m4, m14, m12);
        m13 = sha1_mess(m10, m5, m15, m13);
        m14 = sha1_mess(m11, m6, m0, m14);
        m15 = sha1_mess(m12, m7, m1, m15);

		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( c, d, e, a, b, m0  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( b, c, d, e, a, m1  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( a, b, c, d, e, m2  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( e, a, b, c, d, m3  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( d, e, a, b, c, m4  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( c, d, e, a, b, m5  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( b, c, d, e, a, m6  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( a, b, c, d, e, m7  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( e, a, b, c, d, m8  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( d, e, a, b, c, m9  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( c, d, e, a, b, m10  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( b, c, d, e, a, m11 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( a, b, c, d, e, m12 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( e, a, b, c, d, m13 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( d, e, a, b, c, m14 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( c, d, e, a, b, m15 );
}
{
        m0 = sha1_mess(m13, m8, m2, m0);
        m1 = sha1_mess(m14, m9, m3, m1);
        m2 = sha1_mess(m15, m10, m4, m2);
        m3 = sha1_mess(m0, m11, m5, m3);
        m4 = sha1_mess(m1, m12, m6, m4);
        m5 = sha1_mess(m2, m13, m7, m5);
        m6 = sha1_mess(m3, m14, m8, m6);
        m7 = sha1_mess(m4, m15, m9, m7);
        m8 = sha1_mess(m5, m0, m10, m8);
        m9 = sha1_mess(m6, m1, m11, m9);
        m10 = sha1_mess(m7, m2, m12, m10);
        m11 = sha1_mess(m8, m3, m13, m11);
        m12 = sha1_mess(m9, m4, m14, m12);
        m13 = sha1_mess(m10, m5, m15, m13);
        m14 = sha1_mess(m11, m6, m0, m14);
        m15 = sha1_mess(m12, m7, m1, m15);


		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( b, c, d, e, a, m0  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( a, b, c, d, e, m1  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( e, a, b, c, d, m2  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( d, e, a, b, c, m3  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( c, d, e, a, b, m4  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( b, c, d, e, a, m5  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( a, b, c, d, e, m6  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( e, a, b, c, d, m7  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( d, e, a, b, c, m8  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( c, d, e, a, b, m9  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( b, c, d, e, a, m10 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( a, b, c, d, e, m11 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( e, a, b, c, d, m12 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( d, e, a, b, c, m13 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( c, d, e, a, b, m14 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( b, c, d, e, a, m15 );
}
		ihv[0][threadid] += a; ihv[1][threadid] += b; ihv[2][threadid] += c; ihv[3][threadid] += d; ihv[4][threadid] += e;
		}
	}








__global__ void sha1compress_test2(uint32_t count)
	{
		const uint32_t threadid = threadIdx.x + blockIdx.x*blockDim.x;

		for (unsigned i = 0; i < 5; ++i)
			ihv[i][threadid] = sha1_iv2[i];

		uint32_t a = ihv[0][threadid]; uint32_t b = ihv[1][threadid]; uint32_t c = ihv[2][threadid]; uint32_t d = ihv[3][threadid]; uint32_t e = ihv[4][threadid];
		for (uint32_t i = 0; i < count; ++i)
		{

		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( a, b, c, d, e, me,  0 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( e, a, b, c, d, me,  1 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( d, e, a, b, c, me,  2 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( c, d, e, a, b, me,  3 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( b, c, d, e, a, me,  4 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( a, b, c, d, e, me,  5 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( e, a, b, c, d, me,  6 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( d, e, a, b, c, me,  7 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( c, d, e, a, b, me,  8 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( b, c, d, e, a, me,  9 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( a, b, c, d, e, me, 10 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( e, a, b, c, d, me, 11 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( d, e, a, b, c, me, 12 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( c, d, e, a, b, me, 13 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( b, c, d, e, a, me, 14 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( a, b, c, d, e, me, 15 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( e, a, b, c, d, me, 16 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( d, e, a, b, c, me, 17 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( c, d, e, a, b, me, 18 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEP( b, c, d, e, a, me, 19 );

		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( a, b, c, d, e, me, 20 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( e, a, b, c, d, me, 21 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( d, e, a, b, c, me, 22 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( c, d, e, a, b, me, 23 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( b, c, d, e, a, me, 24 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( a, b, c, d, e, me, 25 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( e, a, b, c, d, me, 26 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( d, e, a, b, c, me, 27 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( c, d, e, a, b, me, 28 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( b, c, d, e, a, me, 29 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( a, b, c, d, e, me, 30 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( e, a, b, c, d, me, 31 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( d, e, a, b, c, me, 32 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( c, d, e, a, b, me, 33 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( b, c, d, e, a, me, 34 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( a, b, c, d, e, me, 35 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( e, a, b, c, d, me, 36 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( d, e, a, b, c, me, 37 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( c, d, e, a, b, me, 38 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEP( b, c, d, e, a, me, 39 );

		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( a, b, c, d, e, me, 40 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( e, a, b, c, d, me, 41 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( d, e, a, b, c, me, 42 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( c, d, e, a, b, me, 43 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( b, c, d, e, a, me, 44 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( a, b, c, d, e, me, 45 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( e, a, b, c, d, me, 46 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( d, e, a, b, c, me, 47 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( c, d, e, a, b, me, 48 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( b, c, d, e, a, me, 49 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( a, b, c, d, e, me, 50 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( e, a, b, c, d, me, 51 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( d, e, a, b, c, me, 52 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( c, d, e, a, b, me, 53 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( b, c, d, e, a, me, 54 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( a, b, c, d, e, me, 55 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( e, a, b, c, d, me, 56 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( d, e, a, b, c, me, 57 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( c, d, e, a, b, me, 58 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEP( b, c, d, e, a, me, 59 );

		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( a, b, c, d, e, me, 60 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( e, a, b, c, d, me, 61 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( d, e, a, b, c, me, 62 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( c, d, e, a, b, me, 63 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( b, c, d, e, a, me, 64 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( a, b, c, d, e, me, 65 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( e, a, b, c, d, me, 66 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( d, e, a, b, c, me, 67 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( c, d, e, a, b, me, 68 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( b, c, d, e, a, me, 69 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( a, b, c, d, e, me, 70 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( e, a, b, c, d, me, 71 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( d, e, a, b, c, me, 72 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( c, d, e, a, b, me, 73 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( b, c, d, e, a, me, 74 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( a, b, c, d, e, me, 75 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( e, a, b, c, d, me, 76 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( d, e, a, b, c, me, 77 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( c, d, e, a, b, me, 78 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEP( b, c, d, e, a, me, 79 );

		}
		ihv[0][threadid] += a; ihv[1][threadid] += b; ihv[2][threadid] += c; ihv[3][threadid] += d; ihv[4][threadid] += e;
	}




__global__ void sha1compress_with_me_test2(uint32_t count)
	{
		const uint32_t threadid = threadIdx.x + blockIdx.x*blockDim.x;

		for (unsigned i = 0; i < 5; ++i)
			ihv[i][threadid] = sha1_iv2[i];

		uint32_t a = ihv[0][threadid]; uint32_t b = ihv[1][threadid]; uint32_t c = ihv[2][threadid]; uint32_t d = ihv[3][threadid]; uint32_t e = ihv[4][threadid];
		for (uint32_t i = 0; i < count; ++i)
		{
		uint32_t m0 = me[0];
		uint32_t m1 = me[1];
		uint32_t m2 = me[2];
		uint32_t m3 = me[3];
		uint32_t m4 = me[4];
		uint32_t m5 = me[5];
		uint32_t m6 = me[6];
		uint32_t m7 = me[7];
		uint32_t m8 = me[8];
		uint32_t m9 = me[9];
		uint32_t m10 = me[10];
		uint32_t m11 = me[11];
		uint32_t m12 = me[12];
		uint32_t m13 = me[13];
		uint32_t m14 = me[14];
		uint32_t m15 = me[15];

		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( a, b, c, d, e, m0 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( e, a, b, c, d, m1 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( d, e, a, b, c, m2 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( c, d, e, a, b, m3 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( b, c, d, e, a, m4 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( a, b, c, d, e, m5 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( e, a, b, c, d, m6 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( d, e, a, b, c, m7 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( c, d, e, a, b, m8 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( b, c, d, e, a, m9 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( a, b, c, d, e, m10 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( e, a, b, c, d, m11 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( d, e, a, b, c, m12 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( c, d, e, a, b, m13 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( b, c, d, e, a, m14 );
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( a, b, c, d, e, m15 );

{
        m0 = sha1_mess(m13, m8, m2, m0);
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( e, a, b, c, d, m0 );
        m1 = sha1_mess(m14, m9, m3, m1);
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( d, e, a, b, c, m1 );
        m2 = sha1_mess(m15, m10, m4, m2);
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( c, d, e, a, b, m2 );
        m3 = sha1_mess(m0, m11, m5, m3);
		HASHCLASH_SHA1COMPRESS_ROUND1_STEPB( b, c, d, e, a, m3 );
        m4 = sha1_mess(m1, m12, m6, m4);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( a, b, c, d, e, m4 );
        m5 = sha1_mess(m2, m13, m7, m5);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( e, a, b, c, d, m5 );
        m6 = sha1_mess(m3, m14, m8, m6);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( d, e, a, b, c, m6 );
        m7 = sha1_mess(m4, m15, m9, m7);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( c, d, e, a, b, m7 );
        m8 = sha1_mess(m5, m0, m10, m8);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( b, c, d, e, a, m8 );
        m9 = sha1_mess(m6, m1, m11, m9);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( a, b, c, d, e, m9 );
        m10 = sha1_mess(m7, m2, m12, m10);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( e, a, b, c, d, m10 );
        m11 = sha1_mess(m8, m3, m13, m11);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( d, e, a, b, c, m11 );
        m12 = sha1_mess(m9, m4, m14, m12);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( c, d, e, a, b, m12 );
        m13 = sha1_mess(m10, m5, m15, m13);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( b, c, d, e, a, m13 );
        m14 = sha1_mess(m11, m6, m0, m14);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( a, b, c, d, e, m14 );
        m15 = sha1_mess(m12, m7, m1, m15);
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( e, a, b, c, d, m15 );
}
{
        m0 = sha1_mess(m13, m8, m2, m0);
        m1 = sha1_mess(m14, m9, m3, m1);
        m2 = sha1_mess(m15, m10, m4, m2);
        m3 = sha1_mess(m0, m11, m5, m3);
        m4 = sha1_mess(m1, m12, m6, m4);
        m5 = sha1_mess(m2, m13, m7, m5);
        m6 = sha1_mess(m3, m14, m8, m6);
        m7 = sha1_mess(m4, m15, m9, m7);
        m8 = sha1_mess(m5, m0, m10, m8);
        m9 = sha1_mess(m6, m1, m11, m9);
        m10 = sha1_mess(m7, m2, m12, m10);
        m11 = sha1_mess(m8, m3, m13, m11);
        m12 = sha1_mess(m9, m4, m14, m12);
        m13 = sha1_mess(m10, m5, m15, m13);
        m14 = sha1_mess(m11, m6, m0, m14);
        m15 = sha1_mess(m12, m7, m1, m15);

		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( d, e, a, b, c, m0 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( c, d, e, a, b, m1 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( b, c, d, e, a, m2 );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( a, b, c, d, e, m3  );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( e, a, b, c, d, m4  );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( d, e, a, b, c, m5  );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( c, d, e, a, b, m6  );
		HASHCLASH_SHA1COMPRESS_ROUND2_STEPB( b, c, d, e, a, m7  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( a, b, c, d, e, m8  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( e, a, b, c, d, m9  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( d, e, a, b, c, m10 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( c, d, e, a, b, m11 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( b, c, d, e, a, m12 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( a, b, c, d, e, m13 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( e, a, b, c, d, m14 );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( d, e, a, b, c, m15 );
}
{
        m0 = sha1_mess(m13, m8, m2, m0);
        m1 = sha1_mess(m14, m9, m3, m1);
        m2 = sha1_mess(m15, m10, m4, m2);
        m3 = sha1_mess(m0, m11, m5, m3);
        m4 = sha1_mess(m1, m12, m6, m4);
        m5 = sha1_mess(m2, m13, m7, m5);
        m6 = sha1_mess(m3, m14, m8, m6);
        m7 = sha1_mess(m4, m15, m9, m7);
        m8 = sha1_mess(m5, m0, m10, m8);
        m9 = sha1_mess(m6, m1, m11, m9);
        m10 = sha1_mess(m7, m2, m12, m10);
        m11 = sha1_mess(m8, m3, m13, m11);
        m12 = sha1_mess(m9, m4, m14, m12);
        m13 = sha1_mess(m10, m5, m15, m13);
        m14 = sha1_mess(m11, m6, m0, m14);
        m15 = sha1_mess(m12, m7, m1, m15);

		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( c, d, e, a, b, m0  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( b, c, d, e, a, m1  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( a, b, c, d, e, m2  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( e, a, b, c, d, m3  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( d, e, a, b, c, m4  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( c, d, e, a, b, m5  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( b, c, d, e, a, m6  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( a, b, c, d, e, m7  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( e, a, b, c, d, m8  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( d, e, a, b, c, m9  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( c, d, e, a, b, m10  );
		HASHCLASH_SHA1COMPRESS_ROUND3_STEPB( b, c, d, e, a, m11 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( a, b, c, d, e, m12 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( e, a, b, c, d, m13 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( d, e, a, b, c, m14 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( c, d, e, a, b, m15 );
}
{
        m0 = sha1_mess(m13, m8, m2, m0);
        m1 = sha1_mess(m14, m9, m3, m1);
        m2 = sha1_mess(m15, m10, m4, m2);
        m3 = sha1_mess(m0, m11, m5, m3);
        m4 = sha1_mess(m1, m12, m6, m4);
        m5 = sha1_mess(m2, m13, m7, m5);
        m6 = sha1_mess(m3, m14, m8, m6);
        m7 = sha1_mess(m4, m15, m9, m7);
        m8 = sha1_mess(m5, m0, m10, m8);
        m9 = sha1_mess(m6, m1, m11, m9);
        m10 = sha1_mess(m7, m2, m12, m10);
        m11 = sha1_mess(m8, m3, m13, m11);
        m12 = sha1_mess(m9, m4, m14, m12);
        m13 = sha1_mess(m10, m5, m15, m13);
        m14 = sha1_mess(m11, m6, m0, m14);
        m15 = sha1_mess(m12, m7, m1, m15);


		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( b, c, d, e, a, m0  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( a, b, c, d, e, m1  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( e, a, b, c, d, m2  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( d, e, a, b, c, m3  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( c, d, e, a, b, m4  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( b, c, d, e, a, m5  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( a, b, c, d, e, m6  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( e, a, b, c, d, m7  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( d, e, a, b, c, m8  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( c, d, e, a, b, m9  );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( b, c, d, e, a, m10 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( a, b, c, d, e, m11 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( e, a, b, c, d, m12 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( d, e, a, b, c, m13 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( c, d, e, a, b, m14 );
		HASHCLASH_SHA1COMPRESS_ROUND4_STEPB( b, c, d, e, a, m15 );
}
		}
		ihv[0][threadid] += a; ihv[1][threadid] += b; ihv[2][threadid] += c; ihv[3][threadid] += d; ihv[4][threadid] += e;
	}












void gpusha1benchmark()
{
	cout << "Using device " << cuda_device << ": " << flush;
	CUDA_ASSERT( cudaSetDevice(cuda_device) );
	cudaDeviceProp prop;
	CUDA_ASSERT( cudaGetDeviceProperties(&prop, cuda_device) );
	cout << prop.name << " (PCI " << hex << setw(2) << setfill('0') << prop.pciBusID << ":" << hex << setw(2) << setfill('0') << prop.pciDeviceID << "." << hex << prop.pciDomainID << dec << ")" << endl;

	timer sw;
	uint32_t count = 1;
	double mintime = 0;
	do {
		count *= 2;
		sw.start();

		sha1compress_test<<<1,32>>>(count);
		cudaDeviceSynchronize();

		mintime = sw.time();
		cout << " (count " << count << ": " << mintime << ")" << flush;
	} while (mintime<1 || count < 16);
	if (count == 0)
		return;
	mintime *= 0.9; // everything below this is an error;

	cout << endl;
	for (int blocks = 13; blocks <= MAXBLOCKS; blocks+=13)
	{
		for (int threads = 64; threads <= MAXTHREADSPERBLOCK; threads *= 2)
		{
			cout << "Starting kernel sha1compress in " << blocks << " block(s) * " << threads << " threads: " << flush;
			sw.start();

			sha1compress_test<<<blocks,threads>>>(count);
			cudaDeviceSynchronize();

			double t = sw.time();
			if (t < mintime)
			{
				cout << "error" << endl;
				break;
			}
			cout << "2^" << (log((double(blocks)*double(threads)*double(count))/t)/log(2.0)) << "#/s" << endl;
		}
	}

	cout << endl;
	for (int blocks = 13; blocks <= MAXBLOCKS; blocks+=13)
	{
		for (int threads = 64; threads <= MAXTHREADSPERBLOCK; threads *= 2)
		{
			cout << "Starting kernel sha1compress_w_me in " << blocks << " block(s) * " << threads << " threads: " << flush;
			sw.start();

			sha1compress_with_me_test<<<blocks,threads>>>(count);
			cudaDeviceSynchronize();

			double t = sw.time();
			if (t < mintime)
			{
				cout << "error" << endl;
				break;
			}
			cout << "2^" << (log((double(blocks)*double(threads)*double(count))/t)/log(2.0)) << "#/s" << endl;
		}
	}


	cout << endl;
	for (int blocks = 13; blocks <= MAXBLOCKS; blocks+=13)
	{
		for (int threads = 64; threads <= MAXTHREADSPERBLOCK; threads *= 2)
		{
			cout << "Starting kernel sha1compress2 in " << blocks << " block(s) * " << threads << " threads: " << flush;
			sw.start();

			sha1compress_test2<<<blocks,threads>>>(count);
			cudaDeviceSynchronize();

			double t = sw.time();
			if (t < mintime)
			{
				cout << "error" << endl;
				break;
			}
			cout << "2^" << (log((double(blocks)*double(threads)*double(count))/t)/log(2.0)) << "#/s" << endl;
		}
	}

	cout << endl;
	for (int blocks = 13; blocks <= MAXBLOCKS; blocks+=13)
	{
		for (int threads = 64; threads <= MAXTHREADSPERBLOCK; threads *= 2)
		{
			cout << "Starting kernel sha1compress_w_me2 in " << blocks << " block(s) * " << threads << " threads: " << flush;
			sw.start();

			sha1compress_with_me_test2<<<blocks,threads>>>(count);
			cudaDeviceSynchronize();

			double t = sw.time();
			if (t < mintime)
			{
				cout << "error" << endl;
				break;
			}
			cout << "2^" << (log((double(blocks)*double(threads)*double(count))/t)/log(2.0)) << "#/s" << endl;
		}
	}

}
