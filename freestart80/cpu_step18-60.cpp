/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
            (C) 2016 Pierre Karpman
            
  This file is part of sha1freestart80 source-code and released under the MIT License
*****/
#include <cstring>

#include "main.hpp"
#include "neutral_bits_packing.hpp"

/*** Bit condition masks for steps Q-4 to Q80, stored on the device in constant memory ***/

// QOFF: value for Q_t is at index QOFF+t in tables below
#define QOFF 4

namespace cpu {
#include "tables.hpp"
}
using namespace cpu;


/* *** SHA1 STEP FUNCTIONS **********************************
 */
inline uint32_t sha1_round1(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e, uint32_t m)
{
	a = rotate_left (a, 5);
	c = rotate_right(c, 2);
	d = rotate_right(d, 2);
	e = rotate_right(e, 2);

	return a + sha1_f1(b, c, d) + e + m + 0x5A827999;
}

inline uint32_t sha1_round2(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e, uint32_t m)
{
	a = rotate_left (a, 5);
	c = rotate_right(c, 2);
	d = rotate_right(d, 2);
	e = rotate_right(e, 2);

	return a + sha1_f2(b, c, d) + e + m + 0x6ED9EBA1 ;
}

#define NEXT_NB(a,mask) { (a) -= 1; (a) &= mask;}


	template<int r> uint32_t sha1_f   (uint32_t b, uint32_t c, uint32_t d); //;{ return 0; }
	template<>      uint32_t sha1_f<0>(uint32_t b, uint32_t c, uint32_t d) { return sha1_f1(b,c,d); }
	template<>      uint32_t sha1_f<1>(uint32_t b, uint32_t c, uint32_t d) { return sha1_f2(b,c,d); }
	template<>      uint32_t sha1_f<2>(uint32_t b, uint32_t c, uint32_t d) { return sha1_f3(b,c,d); }
	template<>      uint32_t sha1_f<3>(uint32_t b, uint32_t c, uint32_t d) { return sha1_f4(b,c,d); }


void cpu_main(std::vector<basesol_t>& basesols)
{
	std::cout << "CPU code is old code from freestart76 and therefore currently disabled." << std::endl;
}

bool verify(basesol_t basesol)
{
	using namespace cpu; // reference cpu tables Qcondmask, ...

	const int firststep = 1, laststep = 16;

	uint32_t main_m1[80];
	uint32_t main_m2[80];
	uint32_t main_Q1[85];
	uint32_t main_Q2[85];
//	uint32_t main_Q1r30[85];
//	uint32_t main_Q2r30[85];

	// basesol: Q12,..,Q17,m5,...,m20
	for (int t = 5; t < 5+16; ++t)
		main_m1[t] = basesol.m[t-5];
	sha1_me_generalised(main_m1, 5);
	for (int t = 12; t <= 17; ++t)
		main_Q1[Qoffset+t] = basesol.Q[t-12];

	// reconstruct state and message words for mainblock: steps t = mainblockoffset+0, ... , mainblockoffset+15
	for (int t = 15; t >= 0; --t)
	{
		sha1_step_bw(t, main_Q1, main_m1);
	}
	for (int t = 17; t < mainblockoffset+16; ++t)
	{
		sha1_step(t, main_Q1, main_m1);
	}

	// [1100] verify stateconditions
	for (int t = firststep - 4; t <= laststep+1; ++t)
	{
			if (0 != (Qcondmask[Qoffset+t] & (
						main_Q1[Qoffset+t]
					^ Qset1mask[Qoffset+t]
					^ ( Qprevmask[Qoffset+t]   & main_Q1[Qoffset+t-1] )
					^ ( Qprevrmask[Qoffset+t]  & rotate_left(main_Q1[Qoffset+t-1],30) )
					^ ( Qprev2rmask[Qoffset+t] & rotate_left(main_Q1[Qoffset+t-2],30) )
					^ ( Qnextmask[Qoffset+t]   & main_Q1[Qoffset+t+1] )
					) ))
			{
					std::cerr << "verify(): Q_" << t << " does not satisfy conditions!" << std::endl;
		return false;
			}
	}

	// [1200] verify message bitrelations
	for (unsigned r = 0; r < msgbitrels_size; ++r)
	{
			bool okay = true;
			uint32_t w = msgbitrels[r][16];
			for (unsigned t = mainblockoffset; t < mainblockoffset+16; ++t)
			{
					if ((t < firststep || t > laststep) && msgbitrels[r][t-mainblockoffset]!=0)
					{
							okay = false;
							break;
					}
					w ^= main_m1[t] & msgbitrels[r][t-mainblockoffset];
			}
			if (okay && 0 != (hc::hw(w)&1) )
			{
					std::cerr << "verify(): bitrelation " << r << " is not satisfied!" << std::endl;
		return false;
			}
	}

	// [1300] verify step computations
	for (int t = firststep; t <= laststep; ++t)
	{
			uint32_t f;
			if (t >=  0 && t<20) f = sha1_f1(main_Q1[Qoffset+t-1],rotate_left(main_Q1[Qoffset+t-2],30),rotate_left(main_Q1[Qoffset+t-3],30));
			if (t >= 20 && t<40) f = sha1_f2(main_Q1[Qoffset+t-1],rotate_left(main_Q1[Qoffset+t-2],30),rotate_left(main_Q1[Qoffset+t-3],30));
			if (t >= 40 && t<60) f = sha1_f3(main_Q1[Qoffset+t-1],rotate_left(main_Q1[Qoffset+t-2],30),rotate_left(main_Q1[Qoffset+t-3],30));
			if (t >= 60 && t<80) f = sha1_f4(main_Q1[Qoffset+t-1],rotate_left(main_Q1[Qoffset+t-2],30),rotate_left(main_Q1[Qoffset+t-3],30));
			uint32_t Qtp1 = rotate_left(main_Q1[Qoffset+t],5) + f + rotate_left(main_Q1[Qoffset+t-4],30) + main_m1[t] + sha1_ac[t/20];
			if (Qtp1 != main_Q1[Qoffset+t+1])
			{
					std::cerr << "verify(): step " << t << " is incorrect!" << std::endl;
		return false;
			}
	}

	if (main_m1[14] & W14NBALLM)
	{
		std::cerr << "neutral bits in W14 are non-zero!" << std::endl;
		return false;
	}
	if (main_m1[15] & W15NBALLM)
	{
		std::cerr << "neutral bits in W15 are non-zero!" << std::endl;
		return false;
	}
	if (main_m1[16] & W16NBALLM)
	{
		std::cerr << "neutral bits in W16 are non-zero!" << std::endl;
		return false;
	}
	if (main_m1[17] & W17NBALLM)
	{
		std::cerr << "neutral bits in W17 are non-zero!" << std::endl;
		return false;
	}
	if (main_m1[18] & W18NBALLM)
	{
		std::cerr << "neutral bits in W18 are non-zero!" << std::endl;
		return false;
	}
	if (main_m1[19] & W19NBALLM)
	{
		std::cerr << "neutral bits in W19 are non-zero!" << std::endl;
		return false;
	}
	if (main_m1[20] & W20NBALLM)
	{
		std::cerr << "neutral bits in W20 are non-zero!" << std::endl;
		return false;
	}
	
	if ((main_m1[10] & Q11BOOMS) || (~main_m1[11] & (Q11BOOMS<<5)) || (~main_m1[15] & (Q11BOOMS>>2)) || (main_Q1[Qoffset+11] & Q11BOOMS))
	{
		std::cerr << "Boom bits for Q11BOOMS bad!" << std::endl;
		return false;
	}

	if ((main_m1[11] & Q12BOOMS) || (~main_m1[12] & (Q12BOOMS<<5)) || (~main_m1[16] & (Q12BOOMS>>2)) || (main_Q1[Qoffset+12] & Q12BOOMS))
	{
		std::cerr << "Boom bits for Q12BOOMS bad!" << std::endl;
		return false;
	}

	if (disable_backwards_filter)
	{
	        return true;
	}

	// verify average overall error probability under neutral bits
	unsigned cnt = 0, badcnt = 0;
	
	uint32_t mbu[80];
	uint32_t Qbu[85];
	memcpy(mbu, main_m1, sizeof(mbu));
	memcpy(Qbu, main_Q1, sizeof(Qbu));
	for (unsigned i = 0; i < 1024; ++i)
	{
        	memcpy(main_m1, mbu, sizeof(mbu));
        	memcpy(main_Q1, Qbu, sizeof(Qbu));
		bool ok = true;
		
		
#if 1
		uint32_t q12booms = xrng128() & Q12BOOMS;
		main_m1[11] ^= q12booms;
		main_m1[12] ^= q12booms << 5;
		main_m1[16] ^= q12booms >> 2;
		main_Q1[Qoffset + 12] ^= q12booms;
#endif
#if 1
		uint32_t q11booms = xrng128() & Q11BOOMS;
		main_m1[10] ^= q11booms;
		main_m1[11] ^= q11booms << 5;
		main_m1[14] ^= (q11booms >> 2) & (1 << 6);
		main_m1[15] ^= q11booms >> 2;
		main_Q1[Qoffset + 11] ^= q11booms;
#endif

#if 1
		main_m1[14] ^= (xrng128() & W14NBALLM);
		main_m1[15] ^= (xrng128() & W15NBALLM);
		main_m1[16] ^= (xrng128() & W16NBALLM);
		main_m1[17] ^= (xrng128() & W17NBALLM);
		main_m1[18] ^= (xrng128() & W18NBALLM);
		main_m1[19] ^= (xrng128() & W19NBALLM);
		main_m1[20] ^= (xrng128() & W20NBALLM);

		//w18[5]:  W15[4]
		main_m1[18] ^= ((0 ^ (main_m1[15] >> 4)) & 1) << 5;

		//w19[13]:  W15[15]  W15[16]  W17[14]  W17[16]  W17[19]
		main_m1[19] ^= ((0 ^ (main_m1[15] >> 15) ^ (main_m1[15] >> 16) ^ (main_m1[17] >> 14) ^ (main_m1[17] >> 16) ^ (main_m1[17] >> 19)) & 1) << 13;

		//w19[15] : W14[10]  W15[12]  W15[13]  W16[13]  W16[14]  W16[15]  W17[11]  W17[13]  W17[16]  W17[18]  W18[12]  W19[10]
		main_m1[19] ^= ((0 ^ (main_m1[14] >> 10) ^ (main_m1[15] >> 12) ^ (main_m1[15] >> 13) ^ (main_m1[16] >> 13) ^ (main_m1[16] >> 14) ^ (main_m1[16] >> 15) ^ (main_m1[17] >> 11) ^ (main_m1[17] >> 13) ^ (main_m1[17] >> 16) ^ (main_m1[17] >> 18) ^ (main_m1[18] >> 12) ^ (main_m1[19] >> 10)) & 1) << 15;

		//w19[16] : W17[17]  W17[19]
		main_m1[19] ^= ((0 ^ (main_m1[17] >> 17) ^ (main_m1[17] >> 19)) & 1) << 16;

		//w19[17] : W16[16]  W17[18]  W17[19]  W18[15]
		main_m1[19] ^= ((0 ^ (main_m1[16] >> 16) ^ (main_m1[17] >> 18) ^ (main_m1[17] >> 19) ^ (main_m1[18] >> 15)) & 1) << 17;

		//w19[18] : W17[19]
		main_m1[19] ^= ((0 ^ (main_m1[17] >> 19)) & 1) << 18;

		//w20[0] : W16[16]  W17[19]  W18[15]
		main_m1[20] ^= ((0 ^ (main_m1[16] >> 16) ^ (main_m1[17] >> 19) ^ (main_m1[18] >> 15)) & 1) << 0;

		//w20[14] : W15[14]  W15[16]  W17[18]  W17[19]
		main_m1[20] ^= ((0 ^ (main_m1[15] >> 14) ^ (main_m1[15] >> 16) ^ (main_m1[17] >> 18) ^ (main_m1[17] >> 19)) & 1) << 14;

		//w20[16] : W15[16]
		main_m1[20] ^= ((0 ^ (main_m1[15] >> 16)) & 1) << 16;

		//w20[17] : W15[16]  W16[13]  W16[15]  W17[16]  W18[12]
		main_m1[20] ^= ((0 ^ (main_m1[15] >> 16) ^ (main_m1[16] >> 13) ^ (main_m1[16] >> 15) ^ (main_m1[17] >> 16) ^ (main_m1[18] >> 12)) & 1) << 17;

		sha1_me_generalised(main_m1, 5);
#endif

#if 0
	// [1200] verify message bitrelations
	for (unsigned r = 0; r < msgbitrels16_size; ++r)
	{
			uint32_t w = msgbitrels16[r][16];
			for (unsigned t = mainblockoffset; t < mainblockoffset+16; ++t)
					w ^= main_m1[t] & msgbitrels16[r][t-mainblockoffset];
			if (0 != (hc::hw(w)&1) )
					std::cerr << "bwfilter: bitrelation16 " << r << " is not satisfied!" << std::endl;
	}
	for (unsigned r = 0; r < msgbitrels80_size; ++r)
	{
			uint32_t w = msgbitrels80[r][80];
			for (unsigned t = 0; t < 80; ++t)
					w ^= main_m1[t] & msgbitrels80[r][t];
			if (0 != (hc::hw(w)&1) )
					std::cerr << "bwfilter: bitrelation80 " << r << " is not satisfied!" << std::endl;
	}
#endif

		for (int t = 5; t >= 0; --t)
		{
			sha1_step_bw(t, main_Q1, main_m1);

			uint32_t Qtm4val = Qset1mask[Qoffset+t-4] ^ (Qnextmask[Qoffset+t-4]&main_Q1[Qoffset+t+1-4]);
			if ( (t-5) >= -4 )
			{
				Qtm4val ^=
					(Qprevmask[Qoffset+t-4]&main_Q1[Qoffset+t-5])
					^ (Qprevrmask[Qoffset+t-4]&rotate_left(main_Q1[Qoffset+t-5],30))
					;
			}
			if ( (t-6) >= -4 )
			{
				Qtm4val ^=
					(Qprev2rmask[Qoffset+t-4]&rotate_left(main_Q1[Qoffset+t-6],30))
					;
			}
	                if (0 != (Qcondmask[Qoffset+t-4] & (main_Q1[Qoffset+t-4]^Qtm4val)))
                	{
				ok = false;
                	}
		}

		for (int t = 0; t < 80; ++t)
		{
			main_m2[t] = main_m1[t] ^ DV_DW[t];
		}
		for (int t = 12-3; t <= 12+1; ++t)
		{
			main_Q2[Qoffset+t] = main_Q1[Qoffset+t] + dQ[Qoffset+t];
		}
		for (int t = 12; t >= 0; --t)
		{
			sha1_step_bw(t, main_Q2, main_m2);
		}
		for (int t = -4; t <= 12+1; ++t)
		{
			if (dQ[Qoffset+t] != main_Q2[Qoffset+t] - main_Q1[Qoffset+t])
			{
				ok = false;
				if (t >= (-4 + 4))
        				cout << "bwfilter: dQ" << t << "bad" << endl; 
			}
		}
#if 0
		for (int t = 13; t <= 16; ++t)
		        sha1_step(t, main_Q1, main_m1);
		for (int t = 13; t <= 16; ++t)
		        sha1_step(t, main_Q2, main_m2);
                for (int t = 14; t <= 16+1; ++t)
                        if (dQ[Qoffset+t] != main_Q2[Qoffset+t] - main_Q1[Qoffset+t])
                        {
                                ok = false;
                                cout << "bwfilter: dQ" << t << "bad" << endl;
                        }
#endif

		++cnt;
		if (!ok)
		{
			++badcnt;
		}
	}
//	cout << " " << badcnt ;
	// dismiss base solution if backwards error probability under neutral bits is more than 0.78125 %
	if ( (float(badcnt)/float(cnt)) > 0.0 ) // 0.05 0.01
	{
		return false;
	}
	return true;
// [1900] end verify()
}
