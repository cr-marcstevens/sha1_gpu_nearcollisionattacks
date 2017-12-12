/*****
  Copyright (C) 2015 Pierre Karpman, INRIA France/Nanyang Technological University Singapore (-2016), CWI (2016/2017), L'Universite Grenoble Alpes (2017-)
            (C) 2015 Thomas Peyrin, Nanyang Technological University Singapore
            (C) 2015 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1_gpu_nearcollisionattacks source-code and released under the MIT License
*****/

#include "main.hpp"

namespace cpu {
#include "tables.hpp"
}
using namespace cpu;

#include "sha1detail.hpp"
#include "rng.hpp"

#include <timer.hpp>

#include <iostream>

using namespace std;
using namespace hashclash;

/*** Bit condition masks for steps Q-4 to Q80, stored on the device in constant memory ***/

// QOFF: value for Q_t is at index QOFF+t in tables below
#define QOFF 4

/// Neutral bits positions //////////////
/// W14: ..................xxxxxxxxx.....
/// W15: ...............x.xxxxxxxxxx.....
/// W16: ...............x..x.xxxxxx......
/// W17: ............x.....xxxx..........
/// W18: ...............xx...............
/// W19: .................x.xxxxxxx......
/// W20: ..................xxx....x.....x
/// W21: ..............xx....x...........
#define W14NBALLM 0x00003FE0
#define W15NBALLM 0x00017FE0
#define W16NBALLM 0x00012FC0
#define W17NBALLM 0x00083C00
#define W18NBALLM 0x00018000
#define W19NBALLM 0x00005FC0
#define W20NBALLM 0x00003841
#define W21NBALLM 0x00030800

/// Per step neutral bits masks /////////
/// Q18
#define W14NBQ18M 0x00003E00 // ..................xxxxx......... (5 of 'em)
#define W15NBQ18M 0x00014000 // ...............x.x.............. (2 of 'em)
/// Q19
#define W14NBQ19M 0x000001E0 // .......................xxxx..... (4 of 'em)
#define W15NBQ19M 0x00003F00 // ..................xxxxxx........ (6 of 'em)
#define W16NBQ19M 0x00012000 // ...............x..x............. (2 of 'em)
#define W17NBQ19M 0x00080000 // ............x................... (1 of 'em)
/// Q20
#define W15NBQ20M 0x000000E0 // ........................xxx..... (3 of 'em)
#define W16NBQ20M 0x00000E00 // ....................xxx......... (3 of 'em)
/// Q21
#define W16NBQ21M 0x000001C0 // .......................xxx...... (3 of 'em)
#define W17NBQ21M 0x00003C00 // ..................xxxx.......... (4 of 'em)
#define W18NBQ21M 0x00018000 // ...............xx............... (2 of 'em)
/// Q23
#define W19NBQ23M 0x00005E00 // .................x.xxxx......... (5 of 'em)
/// Q24
#define W19NBQ24M 0x000000C0 // ........................xx...... (2 of 'em)
#define W20NBQ24M 0x00001800 // ...................xx........... (2 of 'em)
#define W21NBQ24M 0x00030000 // ..............xx................ (2 of 'em)
/// Q25
#define W19NBQ25M 0x00000100 // .......................x........ (1 of 'em)
#define W20NBQ25M 0x00002040 // ..................x......x...... (2 of 'em)
#define W21NBQ25M 0x00000800 // ....................x........... (1 of 'em)
/// Q26
#define W20NBQ26M 0x00000001 // ...............................x (1 of 'em)



/* *** SHA1 STEP FUNCTIONS **********************************
 */
inline uint32_t  sha1_round1(uint32_t  a, uint32_t  b, uint32_t  c, uint32_t  d, uint32_t  e, uint32_t  m)
{
	a = rotate_left (a, 5);
	c = rotate_right(c, 2);
	d = rotate_right(d, 2);
	e = rotate_right(e, 2);

	return a + sha1_f1(b, c, d) + e + m + 0x5A827999;
}

inline uint32_t  sha1_round2(uint32_t  a, uint32_t  b, uint32_t  c, uint32_t  d, uint32_t  e, uint32_t  m)
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

// do everything inside struct to allow for threading later on
struct worker_t {
	uint32_t m1[80];
	uint32_t m2[80];
	uint32_t Q1[85];
	uint32_t Q1r30[85];
	uint32_t Q2[85];

	uint64_t count;
	timer::timer runtime;

#define UPDATE(t)

	worker_t()
	{
		count = 0;
		runtime.start();
	}
	
	void stepcount()
	{
		if (hashclash::hw(++count) == 1)
		{
			cout << "(" << count << "=" << (float(count)/runtime.time()) << "#/s)" << flush;
			if (count >= 16384) exit(0);
		}
	}

	
	template<int t>
	void dosha1step()
	{
		if (t < 40){
		 if (t < 20){
		 	Q1[Qoffset+t+1] = rotate_left(Q1[Qoffset+t],5) + sha1_f1(Q1[Qoffset+t-1],Q1r30[Qoffset+t-2],Q1r30[Qoffset+t-3]) + Q1r30[Qoffset+t-4] + m1[t] + sha1_ac[0];
		 } else {
		 	Q1[Qoffset+t+1] = rotate_left(Q1[Qoffset+t],5) + sha1_f2(Q1[Qoffset+t-1],Q1r30[Qoffset+t-2],Q1r30[Qoffset+t-3]) + Q1r30[Qoffset+t-4] + m1[t] + sha1_ac[1];
		 }
		} else {
		 if (t < 60){
		 	Q1[Qoffset+t+1] = rotate_left(Q1[Qoffset+t],5) + sha1_f3(Q1[Qoffset+t-1],Q1r30[Qoffset+t-2],Q1r30[Qoffset+t-3]) + Q1r30[Qoffset+t-4] + m1[t] + sha1_ac[2];
		 } else {
		 	Q1[Qoffset+t+1] = rotate_left(Q1[Qoffset+t],5) + sha1_f4(Q1[Qoffset+t-1],Q1r30[Qoffset+t-2],Q1r30[Qoffset+t-3]) + Q1r30[Qoffset+t-4] + m1[t] + sha1_ac[3];
		 }
		}
		Q1r30[Qoffset+t+1] = rotate_left(Q1[Qoffset+t+1],30);
	}
	template<int t>
	void domsgexp()
	{
		m1[t]=rotate_left(m1[t-3] ^ m1[t-8] ^ m1[t-14] ^ m1[t-16], 1);
	}
	template<int t>
	bool isQokay()
	{
		uint32_t Qval = Qset1mask[Qoffset+t]
			^ (Qprevmask[Qoffset+t]&Q1[Qoffset+t-1]) 
			^ (Qprevrmask[Qoffset+t]&Q1r30[Qoffset+t-1]) 
			^ (Qprev2rmask[Qoffset+t]&Q1r30[Qoffset+t-2])
			;
		return 0 == ( (Q1[Qoffset+t]^Qval) & Qcondmask[Qoffset+t] );
	}
	
	void checkcoll()
	{
		stepcount();
/*
		for (int i = 6; i < 6+16; ++i)
			m2[i] = m1[i];
		sha1_me_generalised(m2,6);
		for (int t = -4; t <= 0; ++t)
			Q2[Qoffset+16+t] = Q1[Qoffset+16+t];
		for (int t = 16; t < 25; ++t)
		{
			sha1_step(t, Q2, m2);
			if (Q2[Qoffset+t+1] != Q1[Qoffset+t+1])
				cerr << "bad Q" << (t+1) << endl;			
		}
*/				
	}

	void stepQ25()
	{
		UPDATE(25);
				
		uint32_t q20bu = Q1[Qoffset+20];
		uint32_t q21bu = Q1[Qoffset+21];
		uint32_t q22bu = Q1[Qoffset+22];
		uint32_t q23bu = Q1[Qoffset+23];
		uint32_t q24bu = Q1[Qoffset+24];
		uint32_t m19bu = m1[19];
		uint32_t m20bu = m1[20];
		uint32_t m21bu = m1[21];
		
		uint32_t m19nb = 0;
		do
		{
			NEXT_NB(m19nb, W19NBQ25M);

			m1[19] = m19bu | m19nb;
			Q1[Qoffset+20] = q20bu + m19nb;
			if ( (Q1[Qoffset+20]^q20bu) & Qcondmask[Qoffset+20] ) continue;
			Q1r30[Qoffset+20] = rotate_left(Q1[Qoffset+20],30);

					domsgexp<22>();

			uint32_t q21precomp = q21bu - rotate_left(q20bu,5) + rotate_left(Q1[Qoffset+20],5);
			uint32_t q22precomp = sha1_f2(Q1[Qoffset+20],Q1r30[Qoffset+19],Q1r30[Qoffset+18]) + Q1r30[Qoffset+17] + sha1_ac[1];

			uint32_t m20nb = 0;
			do
			{
				NEXT_NB(m20nb, W20NBQ25M);

				m1[20] = m20bu | m20nb;
				Q1[Qoffset+21] = q21precomp + m20nb;

				if ( (Q1[Qoffset+21]^q21bu) & Qcondmask[Qoffset+21] ) continue;
				Q1r30[Qoffset+21] = rotate_left(Q1[Qoffset+21],30);

				uint32_t q22precomp2 = q22precomp + rotate_left(Q1[Qoffset+21],5);
				uint32_t q23precomp = sha1_f2(Q1[Qoffset+21],Q1r30[Qoffset+20],Q1r30[Qoffset+19]) + Q1r30[Qoffset+18] + sha1_ac[1] + m1[22];

					domsgexp<23>();

				uint32_t m21nb = 0;
				do
				{
				  bool Q25okay = false;
				  do {
					NEXT_NB(m21nb, W21NBQ25M);

					m1[21] = m21bu | m21nb;
					Q1[Qoffset+22] = q22precomp2 + m1[21];
					Q1r30[Qoffset+22] = rotate_left(Q1[Qoffset+22],30);
					Q1[Qoffset+23] = q23precomp + rotate_left(Q1[Qoffset+22],5);
					Q1r30[Qoffset+23] = rotate_left(Q1[Qoffset+23],30);


					dosha1step<23>();

					domsgexp<24>();
					dosha1step<24>();
					if (!isQokay<25>()) continue;
					Q25okay = true;

					domsgexp<25>();
					dosha1step<25>();
					if (!isQokay<26>()) continue;

					domsgexp<26>();
					dosha1step<26>();
					if (!isQokay<27>()) continue;

					domsgexp<27>();
					dosha1step<27>();
					if (!isQokay<28>()) continue;

					domsgexp<28>();
					dosha1step<28>();
					if (!isQokay<29>()) continue;

					domsgexp<29>();
					dosha1step<29>();
					if (!isQokay<30>()) continue;

					domsgexp<30>();
					dosha1step<30>();
					if (!isQokay<31>()) continue;

					domsgexp<31>();
					dosha1step<31>();
					if (!isQokay<32>()) continue;

					domsgexp<32>();
					dosha1step<32>();
					if (!isQokay<33>()) continue;

					if (!isQokay<22>()) continue;
					if (!isQokay<23>()) continue;
					if (!isQokay<24>()) continue;

					checkcoll();
				  } while (false);

				  if (Q25okay)
				  {
				    do {
					m1[20] ^= W20NBQ26M; // 1 bit in W20NBQ26M
					Q1[Qoffset+21] += W20NBQ26M;
					Q1r30[Qoffset+21] = rotate_left(Q1[Qoffset+21],30);
					Q1[Qoffset+22] = q22precomp + rotate_left(Q1[Qoffset+21],5) + m1[21];
					Q1r30[Qoffset+22] = rotate_left(Q1[Qoffset+22],30);
					domsgexp<23>();

					dosha1step<22>();
					dosha1step<23>();

					domsgexp<24>();
					dosha1step<24>();

					domsgexp<25>();
					dosha1step<25>();
					if (!isQokay<26>()) continue;

					domsgexp<26>();
					dosha1step<26>();
					if (!isQokay<27>()) continue;

					domsgexp<27>();
					dosha1step<27>();
					if (!isQokay<28>()) continue;

					domsgexp<28>();
					dosha1step<28>();
					if (!isQokay<29>()) continue;

					domsgexp<29>();
					dosha1step<29>();
					if (!isQokay<30>()) continue;

					domsgexp<30>();
					dosha1step<30>();
					if (!isQokay<31>()) continue;

					domsgexp<31>();
					dosha1step<31>();
					if (!isQokay<32>()) continue;

					domsgexp<32>();
					dosha1step<32>();
					if (!isQokay<33>()) continue;

					if (!isQokay<22>()) continue;
					if (!isQokay<23>()) continue;
					if (!isQokay<24>()) continue;

					checkcoll();

				    } while (false);
				    m1[20] ^= W20NBQ26M;
				    Q1[Qoffset+21] -= W20NBQ26M;
				    Q1r30[Qoffset+21] = rotate_left(Q1[Qoffset+21],30);
				    Q1[Qoffset+22] = q22precomp + rotate_left(Q1[Qoffset+21],5) + m1[21];
				    Q1r30[Qoffset+22] = rotate_left(Q1[Qoffset+22],30);
				    domsgexp<23>();
				  }

				} while (m21nb != 0);
			} while (m20nb != 0);
		} while (m19nb != 0);

		Q1[Qoffset+20] = q20bu;
		Q1[Qoffset+21] = q21bu;
//		Q1[Qoffset+22] = q22bu;
//		Q1[Qoffset+23] = q23bu;
//		Q1[Qoffset+24] = q24bu;
		Q1r30[Qoffset+20] = rotate_left(Q1[Qoffset+20],30);
		Q1r30[Qoffset+21] = rotate_left(Q1[Qoffset+21],30);
//		Q1r30[Qoffset+22] = rotate_left(Q1[Qoffset+22],30);
//		Q1r30[Qoffset+23] = rotate_left(Q1[Qoffset+23],30);
//		Q1r30[Qoffset+24] = rotate_left(Q1[Qoffset+24],30);
		m1[19] = m19bu;
		m1[20] = m20bu;
//		m1[21] = m21bu;
	}


	void stepQ24()
	{
		UPDATE(24);
		
		uint32_t q20bu = Q1[Qoffset+20];
		uint32_t q21bu = Q1[Qoffset+21];
		uint32_t q22bu = Q1[Qoffset+22];
		uint32_t q23bu = Q1[Qoffset+23];
		uint32_t m19bu = m1[19];
		uint32_t m20bu = m1[20];
		uint32_t m21bu = m1[21];
		
		uint32_t m19nb = 0;
		do
		{
			NEXT_NB(m19nb, W19NBQ24M);

			m1[19] = m19bu | m19nb;
			Q1[Qoffset+20] = q20bu + m19nb;
			if ( (Q1[Qoffset+20]^q20bu) & Qcondmask[Qoffset+20] ) continue;
			Q1r30[Qoffset+20] = rotate_left(Q1[Qoffset+20],30);

					domsgexp<22>();
			
			uint32_t m20nb = 0;
			do
			{
				NEXT_NB(m20nb, W20NBQ24M);

				m1[20] = m20bu | m20nb;
				dosha1step<20>();
				if ( (Q1[Qoffset+21]^q21bu) & Qcondmask[Qoffset+21] ) continue;

					domsgexp<23>();

				uint32_t m21nb = 0;
				do
				{
					NEXT_NB(m21nb, W21NBQ24M);

					m1[21] = m21bu | m21nb;
					dosha1step<21>();
					
					dosha1step<22>();
					dosha1step<23>();
					if (!isQokay<24>()) continue;

					if ( (Q1[Qoffset+22]^q22bu) & Qcondmask[Qoffset+22] ) continue;
					if ( (Q1[Qoffset+23]^q23bu) & Qcondmask[Qoffset+23] ) continue;
			
					stepQ25();

				} while (m21nb != 0);
			} while (m20nb != 0);
		} while (m19nb != 0);

		Q1[Qoffset+20] = q20bu;
		Q1[Qoffset+21] = q21bu;
		Q1[Qoffset+22] = q22bu;
		Q1[Qoffset+23] = q23bu;
		Q1r30[Qoffset+20] = rotate_left(Q1[Qoffset+20],30);
		Q1r30[Qoffset+21] = rotate_left(Q1[Qoffset+21],30);
		Q1r30[Qoffset+22] = rotate_left(Q1[Qoffset+22],30);
		Q1r30[Qoffset+23] = rotate_left(Q1[Qoffset+23],30);
		m1[19] = m19bu;
		m1[20] = m20bu;
		m1[21] = m21bu;
	}
	
	void stepQ23()
	{
		UPDATE(23);
		
		uint32_t q20bu = Q1[Qoffset+20];
		uint32_t q21bu = Q1[Qoffset+21];
		uint32_t q22bu = Q1[Qoffset+22];
		uint32_t m19bu = m1[19];
		
		uint32_t m19nb = 0;
		do
		{
			NEXT_NB(m19nb, W19NBQ23M);

			m1[19] = m19bu | m19nb;
			Q1[Qoffset+20] = q20bu + m19nb;
			Q1r30[Qoffset+20] = rotate_left(Q1[Qoffset+20],30);
			
			Q1[Qoffset+21] = q21bu + (rotate_left(Q1[Qoffset+20],5) - rotate_left(q20bu,5));
			Q1r30[Qoffset+21] = rotate_left(Q1[Qoffset+21],30);
			
			dosha1step<21>();
			
			domsgexp<22>();
			dosha1step<22>();
			if (!isQokay<23>()) continue;

			if ( (Q1[Qoffset+20]^q20bu) & Qcondmask[Qoffset+20] ) continue;
			if ( (Q1[Qoffset+21]^q21bu) & Qcondmask[Qoffset+21] ) continue;
			if ( (Q1[Qoffset+22]^q22bu) & Qcondmask[Qoffset+22] ) continue;
			
			stepQ24();
			
		} while (m19nb != 0);

		Q1[Qoffset+20] = q20bu;
		Q1[Qoffset+21] = q21bu;
		Q1[Qoffset+22] = q22bu;
		Q1r30[Qoffset+20] = rotate_left(Q1[Qoffset+20],30);
		Q1r30[Qoffset+21] = rotate_left(Q1[Qoffset+21],30);
		Q1r30[Qoffset+22] = rotate_left(Q1[Qoffset+22],30);
		m1[19] = m19bu;
	}

	void stepQ21()
	{
		UPDATE(21);
		uint32_t q17bu = Q1[Qoffset+17];
		uint32_t q18bu = Q1[Qoffset+18];
		uint32_t q19bu = Q1[Qoffset+19];
		uint32_t q20bu = Q1[Qoffset+20];
		uint32_t m16bu = m1[16];
		uint32_t m17bu = m1[17];
		uint32_t m18bu = m1[18];
		
		uint32_t m16nb = 0;
		do
		{
			NEXT_NB(m16nb, W16NBQ21M);
			m1[16] = m16bu | m16nb;

			dosha1step<16>();
			if ( (Q1[Qoffset+17]^q17bu) & Qcondmask[Qoffset+17] ) continue;
			
			uint32_t m17nb = 0;
			do
			{
				NEXT_NB(m17nb, W17NBQ21M);
				m1[17] = m17bu | m17nb;

				dosha1step<17>();
				if ( (Q1[Qoffset+18]^q18bu) & Qcondmask[Qoffset+18] ) continue;
			
				uint32_t m18nb = 0;
				do
				{
					NEXT_NB(m18nb, W18NBQ21M);
					m1[18] = m18bu | m18nb;
					
					dosha1step<18>();
					dosha1step<19>();

					dosha1step<20>();
					if (!isQokay<21>()) continue;

					dosha1step<21>();
					if (!isQokay<22>()) continue;

					if ( (Q1[Qoffset+19]^q19bu) & Qcondmask[Qoffset+19] ) continue;
					if ( (Q1[Qoffset+20]^q20bu) & Qcondmask[Qoffset+20] ) continue;
			
					stepQ23();
			
				} while (m18nb != 0);
			} while (m17nb != 0);
		} while (m16nb != 0);

		Q1[Qoffset+17] = q17bu;
		Q1[Qoffset+18] = q18bu;
		Q1[Qoffset+19] = q19bu;
		Q1[Qoffset+20] = q20bu;
		Q1r30[Qoffset+17] = rotate_left(Q1[Qoffset+17],30);
		Q1r30[Qoffset+18] = rotate_left(Q1[Qoffset+18],30);
		Q1r30[Qoffset+19] = rotate_left(Q1[Qoffset+19],30);
		Q1r30[Qoffset+20] = rotate_left(Q1[Qoffset+20],30);
		m1[16] = m16bu;
		m1[17] = m17bu;
		m1[18] = m18bu;
	}

	void stepQ20()
	{
		UPDATE(20);
		uint32_t q16bu = Q1[Qoffset+16];
		uint32_t q17bu = Q1[Qoffset+17];
		uint32_t q18bu = Q1[Qoffset+18];
		uint32_t q19bu = Q1[Qoffset+19];
		uint32_t m15bu = m1[15];
		uint32_t m16bu = m1[16];

		uint32_t m15nb = 0;
		do
		{
			NEXT_NB(m15nb, W15NBQ20M);
			m1[15] = m15bu | m15nb;

			Q1[Qoffset+16] = q16bu + m15nb;	
			if (!isQokay<16>()) continue;
			Q1r30[Qoffset+16] = rotate_left(Q1[Qoffset+16],30);

			uint32_t m16nb = 0;
			do
			{
				NEXT_NB(m16nb, W16NBQ20M);
				m1[16] = m16bu | m16nb;

				dosha1step<16>();
				dosha1step<17>();
				dosha1step<18>();
				dosha1step<19>();
				if (!isQokay<20>()) continue;
				if (!isQokay<19>()) continue;
				if (!isQokay<18>()) continue;
				if (!isQokay<17>()) continue;

				stepQ21();

			} while (m16nb != 0);
		} while (m15nb != 0);

		Q1[Qoffset+16] = q16bu;
		Q1[Qoffset+17] = q17bu;
		Q1[Qoffset+18] = q18bu;
		Q1r30[Qoffset+16] = rotate_left(Q1[Qoffset+16],30);
		Q1r30[Qoffset+17] = rotate_left(Q1[Qoffset+17],30);
		Q1r30[Qoffset+18] = rotate_left(Q1[Qoffset+18],30);
		m1[15] = m15bu;
		m1[16] = m16bu;
	}

	void stepQ19()
	{
		UPDATE(19);
		uint32_t q15bu = Q1[Qoffset+15];
		uint32_t q16bu = Q1[Qoffset+16];
		uint32_t q17bu = Q1[Qoffset+17];
		uint32_t q18bu = Q1[Qoffset+18];
		uint32_t m14bu = m1[14];
		uint32_t m15bu = m1[15];
		uint32_t m16bu = m1[16];
		uint32_t m17bu = m1[17];

		uint32_t m14nb = 0;
		do
		{
			NEXT_NB(m14nb, W14NBQ19M);
			m1[14] = m14bu | m14nb;

			Q1[Qoffset+15] = q15bu + m14nb;
			Q1r30[Qoffset+15] = rotate_left(Q1[Qoffset+15],30);

			uint32_t m15nb = 0;
			do
			{
				NEXT_NB(m15nb, W15NBQ19M);
				m1[15] = m15bu | m15nb;

				Q1[Qoffset+16] = q16bu + (rotate_left(Q1[Qoffset+15],5) - rotate_left(q15bu,5)) + m15nb;
				if (!isQokay<16>()) continue;
				Q1r30[Qoffset+16] = rotate_left(Q1[Qoffset+16],30);

				uint32_t m16nb = 0;
				do
				{
					NEXT_NB(m16nb, W16NBQ19M);
					m1[16] = m16bu | m16nb;

					dosha1step<16>();
					if (!isQokay<17>()) continue;

					uint32_t m17nb = 0;
					do
					{
						NEXT_NB(m17nb, W17NBQ19M);
						m1[17] = m17bu | m17nb;

						dosha1step<17>();
						dosha1step<18>();
						if (!isQokay<19>()) continue;
						if (!isQokay<18>()) continue;

						stepQ20();

					} while (m17nb != 0);
				} while (m16nb != 0);
			} while (m15nb != 0);
		} while (m14nb != 0);

		Q1[Qoffset+15] = q15bu;
		Q1[Qoffset+16] = q16bu;
		Q1[Qoffset+17] = q17bu;
		Q1r30[Qoffset+15] = rotate_left(Q1[Qoffset+15],30);
		Q1r30[Qoffset+16] = rotate_left(Q1[Qoffset+16],30);
		Q1r30[Qoffset+17] = rotate_left(Q1[Qoffset+17],30);
		m1[14] = m14bu;
		m1[15] = m15bu;
		m1[16] = m16bu;
		m1[17] = m17bu;
	}

	void stepQ18(const basesol_t& basesol)
	{
		UPDATE(18);
		for (int t = 12; t <= 17; ++t)
		{
			Q1[Qoffset+t] = basesol.Q[t-12];
			Q1r30[Qoffset+t] = rotate_left(Q1[Qoffset+t],30);
		}
		for (int t = 6; t < 6+16; ++t)
			m1[t] = basesol.m[t-6];
		sha1_me_generalised(m1, 6);
		
		if (!isQokay<14>()) cerr << "badQ14" << endl;
		if (!isQokay<15>()) cerr << "badQ15" << endl;
		if (!isQokay<16>()) cerr << "badQ16" << endl;
		if (!isQokay<17>()) cerr << "badQ17" << endl;

		uint32_t q15bu = Q1[Qoffset+15];
		uint32_t q16bu = Q1[Qoffset+16];
		uint32_t q17bu = Q1[Qoffset+17];
		uint32_t m14bu = m1[14];
		uint32_t m15bu = m1[15];
		
		uint32_t m14nb = 0;
		do
		{
			NEXT_NB(m14nb, W14NBQ18M);
			m1[14] = m14bu | m14nb;

			Q1[Qoffset+15] = q15bu + m14nb;
			Q1r30[Qoffset+15] = rotate_left(Q1[Qoffset+15],30);
			
			uint32_t m15nb = 0;
			do
			{
				NEXT_NB(m15nb, W15NBQ18M);
				m1[15] = m15bu | m15nb;

				Q1[Qoffset+16] = q16bu + (rotate_left(Q1[Qoffset+15],5) - rotate_left(q15bu,5)) + m15nb;
				Q1r30[Qoffset+16] = rotate_left(Q1[Qoffset+16],30);

				dosha1step<16>();
				dosha1step<17>();
				if (!isQokay<18>()) continue;
				if (!isQokay<17>()) continue;
				if (!isQokay<16>()) continue;

				stepQ19();

			} while (m15nb != 0);
		} while (m14nb != 0);
	}

};

void cpu_main(std::vector<basesol_t>& basesols)
{
	worker_t worker;
	for (size_t i = 0; i < basesols.size(); ++i)
	{
		if (!verify(basesols[i]))
		{
			cout << "!" << flush;
			continue;
		}
		cout << "." << flush;
		worker.stepQ18(basesols[i]);
	}
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

	// basesol: Q12,..,Q17,m6,...,m21
	for (int t = 6; t < 6+16; ++t)
		main_m1[t] = basesol.m[t-6];
	sha1_me_generalised(main_m1, 6);
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
                        std::cerr << "Q_" << t << " does not satisfy conditions!" << std::endl;
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
                        if ((t < firststep | t > laststep) && msgbitrels[r][t-mainblockoffset]!=0)
                        {
                                okay = false;
                                break;
                        }
                        w ^= main_m1[t] & msgbitrels[r][t-mainblockoffset];
                }
                if (okay && 0 != (hashclash::hw(w)&1) )
                {
                        std::cerr << "bitrelation " << r << " is not satisfied!" << std::endl;
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
                        std::cerr << "step " << t << " is incorrect!" << std::endl;
			return false;
                }
        }

	// verify neutral bits are initially set 0
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
	if (main_m1[21] & W21NBALLM)
	{
		std::cerr << "neutral bits in W21 are non-zero!" << std::endl;
		return false;
	}
	
	if (disable_backwards_filter)
	{
	        return true;
	}

	// verify average overall error probability under neutral bits
	unsigned cnt = 0, badcnt = 0;
	for (unsigned i = 0; i < 256; ++i)
	{
		bool ok = true;
		main_m1[14] ^= (xrng128() & W14NBALLM);
		main_m1[15] ^= (xrng128() & W15NBALLM);
		main_m1[16] ^= (xrng128() & W16NBALLM);
		main_m1[17] ^= (xrng128() & W17NBALLM);
		main_m1[18] ^= (xrng128() & W18NBALLM);
		main_m1[19] ^= (xrng128() & W19NBALLM);
		main_m1[20] ^= (xrng128() & W20NBALLM);
		main_m1[21] ^= (xrng128() & W21NBALLM);

		main_m1[20] ^= ( ( (main_m1[15]>>14) ^ (main_m1[15]>>16) ^ (main_m1[16]>>16) ^ (main_m1[18]>>15) ) & 1) << 14;
		main_m1[20] ^= ( ( (main_m1[15]>>16) ) & 1) << 16;

		sha1_me_generalised(main_m1, 6);

		for (int t = 6; t >= 0; --t)
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
		for (int t = 14-5; t <= 14; ++t)
		{
			main_Q2[Qoffset+t] = main_Q1[Qoffset+t] + dQ[Qoffset+t];
		}
		for (int t = 13; t >= 0; --t)
		{
			sha1_step_bw(t, main_Q2, main_m2);
		}
		for (int t = -4; t <= 0; ++t)
		{
			if (dQ[Qoffset+t] != main_Q2[Qoffset+t] - main_Q1[Qoffset+t])
			{
				ok = false;
			}
			else
			{
//				cout << " dQ" << t << "OK" << flush; 
			}
		}

		++cnt;
		if (!ok)
		{
			++badcnt;
		}
	}
	// dismiss base solution if backwards error probability under neutral bits is more than 0.78125 %
	if (badcnt >= 2)
	{
		return false;
	}
	return true;
// [1900] end verify()
}
