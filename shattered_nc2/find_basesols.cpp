/******
   basesolgen.cpp:
   modified generated_code.cpp based on:
    org-sys with Q7,Q10 boomerangs with Q14-Q20 neutral bits
   with following modifications:
   
   * removed step 13,14,15 (Q14,Q15,Q16) condition verification
   
   * removed step 13,14,15 auto fix condition
   
   * step15: verify(0,15,13): only verify Q-conds up to Q13, not Q16
   
   * verify modified:
     * fixed bug: firststep&laststep should be int not unsigned
     * added lastQ (def 80) param to verify Q-conds only up to Q_(lastQ) and not Q_(laststep+1)
     * replaced hardcoded main_Q1, main_m1 with parameters mQ1, mm1 with default value main_Q1, main_m1
     
********/


// [-100] preprocessor defines
#ifndef PERF_ENTER_STEP
#define PERF_ENTER_STEP(s) {}
#endif

#include "main.hpp"

#include "sha1detail.hpp"
#include "rng.hpp"

// [-90] include files system
#include <cstdlib>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
//#include <array>
#include <cstring>
#include <cmath>

using namespace std;
using namespace hc;

inline uint32_t rng() { return xrng128(); }

																																																																																																				// [-70] typedefs
// [-60] global tables const
// mainblockoffset: the main block of the first 16 steps will consist of steps 0+mainblockoffset,...,15+mainblockoffset (not necessarily in that order)
const uint32_t mainblockoffset = 0;
// reversesteps: the first reversesteps # of steps in main block will be done in reverse order
const uint32_t reversesteps = 0;
// i.e., steps will be computed in this order:
//  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

// Qoffset: value for Q_t is at index Qoffset+t in tables below
const int Qoffset = 4;

// Qcondmask  : Qt[b]: 0-bit = free, 1-bit = has condition
const uint32_t Qcondmask  [85] = { /*0*/ 0xffffffff, /*1*/ 0xffffffff, /*2*/ 0xffffffff, /*3*/ 0xffffffff, /*4*/ 0xffffffff, /*5*/ 0xffefe7f7, /*6*/ 0xffffffdd, /*7*/ 0xffffff7f, /*8*/ 0x7ffff7f7, /*9*/ 0x7ffffcc5, /*10*/ 0xf5fffdf7, /*11*/ 0x7c0001fd, /*12*/ 0x14000075, /*13*/ 0x50000259, /*14*/ 0xd4000088, /*15*/ 0x70000020, /*16*/ 0x88000022, /*17*/ 0x88000000, /*18*/ 0x20000002, /*19*/ 0xa8000003, /*20*/ 0xe8000002, /*21*/ 0x90000002, /*22*/ 0xb0000000, /*23*/ 0x88000000, /*24*/ 0xa8000000, /*25*/ 0x08000000, /*26*/ 0x28000000, /*27*/ 0x88000001, /*28*/ 0x28000000, /*29*/ 0x18000000, /*30*/ 0x08000000, /*31*/ 0x00000000, /*32*/ 0x00000000, /*33*/ 0x00000000, /*34*/ 0x00000000, /*35*/ 0x00000000, /*36*/ 0x00000000, /*37*/ 0x00000000, /*38*/ 0x00000000, /*39*/ 0x00000000, /*40*/ 0x00000000, /*41*/ 0x00000000, /*42*/ 0x00000000, /*43*/ 0x00000000, /*44*/ 0x00000000, /*45*/ 0x00000000, /*46*/ 0x00000000, /*47*/ 0x00000000, /*48*/ 0x00000000, /*49*/ 0x00000000, /*50*/ 0x00000000, /*51*/ 0x00000000, /*52*/ 0x00000000, /*53*/ 0x00000000, /*54*/ 0x00000000, /*55*/ 0x00000000, /*56*/ 0x00000000, /*57*/ 0x00000000, /*58*/ 0x00000000, /*59*/ 0x00000000, /*60*/ 0x00000000, /*61*/ 0x00000000, /*62*/ 0x00000000, /*63*/ 0x00000000, /*64*/ 0x00000000, /*65*/ 0x00000000, /*66*/ 0x00000000, /*67*/ 0x00000000, /*68*/ 0x00000000, /*69*/ 0x00000000, /*70*/ 0x00000000, /*71*/ 0x00000000, /*72*/ 0x00000000, /*73*/ 0x00000000, /*74*/ 0x00000000, /*75*/ 0x00000000, /*76*/ 0x00000000, /*77*/ 0x00000000, /*78*/ 0x00000000, /*79*/ 0x00000000, /*80*/ 0x00000000, /*81*/ 0x00000000, /*82*/ 0x00000000, /*83*/ 0x00000000, /*84*/ 0x00000000 };

// Qset1mask  : Qt[b]: 0-bit = free|plus|zero|prev|prevr|prev2r|next, 1-bit = minus|one|prevn|prevrn|prev2rn
const uint32_t Qset1mask  [85] = { /*0*/ 0xce2969ef, /*1*/ 0x7b1facd1, /*2*/ 0xaf216457, /*3*/ 0xffed5352, /*4*/ 0x8d64d617, /*5*/ 0x16c9a022, /*6*/ 0x567a5ecc, /*7*/ 0xa2411708, /*8*/ 0x1bfff3b7, /*9*/ 0x06322404, /*10*/ 0x41bfc402, /*11*/ 0x40000004, /*12*/ 0x00000024, /*13*/ 0x50000059, /*14*/ 0x54000000, /*15*/ 0x10000000, /*16*/ 0x80000022, /*17*/ 0x08000000, /*18*/ 0x00000002, /*19*/ 0xa8000000, /*20*/ 0x80000002, /*21*/ 0x90000000, /*22*/ 0xa0000000, /*23*/ 0x88000000, /*24*/ 0xa0000000, /*25*/ 0x00000000, /*26*/ 0x28000000, /*27*/ 0x80000001, /*28*/ 0x20000000, /*29*/ 0x18000000, /*30*/ 0x00000000, /*31*/ 0x00000000, /*32*/ 0x00000000, /*33*/ 0x00000000, /*34*/ 0x00000000, /*35*/ 0x00000000, /*36*/ 0x00000000, /*37*/ 0x00000000, /*38*/ 0x00000000, /*39*/ 0x00000000, /*40*/ 0x00000000, /*41*/ 0x00000000, /*42*/ 0x00000000, /*43*/ 0x00000000, /*44*/ 0x00000000, /*45*/ 0x00000000, /*46*/ 0x00000000, /*47*/ 0x00000000, /*48*/ 0x00000000, /*49*/ 0x00000000, /*50*/ 0x00000000, /*51*/ 0x00000000, /*52*/ 0x00000000, /*53*/ 0x00000000, /*54*/ 0x00000000, /*55*/ 0x00000000, /*56*/ 0x00000000, /*57*/ 0x00000000, /*58*/ 0x00000000, /*59*/ 0x00000000, /*60*/ 0x00000000, /*61*/ 0x00000000, /*62*/ 0x00000000, /*63*/ 0x00000000, /*64*/ 0x00000000, /*65*/ 0x00000000, /*66*/ 0x00000000, /*67*/ 0x00000000, /*68*/ 0x00000000, /*69*/ 0x00000000, /*70*/ 0x00000000, /*71*/ 0x00000000, /*72*/ 0x00000000, /*73*/ 0x00000000, /*74*/ 0x00000000, /*75*/ 0x00000000, /*76*/ 0x00000000, /*77*/ 0x00000000, /*78*/ 0x00000000, /*79*/ 0x00000000, /*80*/ 0x00000000, /*81*/ 0x00000000, /*82*/ 0x00000000, /*83*/ 0x00000000, /*84*/ 0x00000000 };

// Qprevmask  : Qt[b]: 1-bit = prev|prevn, 0-bit otherwise
const uint32_t Qprevmask  [85] = { /*0*/ 0x00000000, /*1*/ 0x00000000, /*2*/ 0x00000000, /*3*/ 0x00000000, /*4*/ 0x00000000, /*5*/ 0x00000000, /*6*/ 0x00000000, /*7*/ 0x00000000, /*8*/ 0x00000000, /*9*/ 0x00000000, /*10*/ 0x00000100, /*11*/ 0x00000000, /*12*/ 0x00000000, /*13*/ 0x00000200, /*14*/ 0x80000000, /*15*/ 0x00000000, /*16*/ 0x00000002, /*17*/ 0x00000000, /*18*/ 0x00000002, /*19*/ 0x00000003, /*20*/ 0x00000002, /*21*/ 0x00000002, /*22*/ 0x00000000, /*23*/ 0x00000000, /*24*/ 0x00000000, /*25*/ 0x08000000, /*26*/ 0x00000000, /*27*/ 0x00000001, /*28*/ 0x00000000, /*29*/ 0x00000000, /*30*/ 0x00000000, /*31*/ 0x00000000, /*32*/ 0x00000000, /*33*/ 0x00000000, /*34*/ 0x00000000, /*35*/ 0x00000000, /*36*/ 0x00000000, /*37*/ 0x00000000, /*38*/ 0x00000000, /*39*/ 0x00000000, /*40*/ 0x00000000, /*41*/ 0x00000000, /*42*/ 0x00000000, /*43*/ 0x00000000, /*44*/ 0x00000000, /*45*/ 0x00000000, /*46*/ 0x00000000, /*47*/ 0x00000000, /*48*/ 0x00000000, /*49*/ 0x00000000, /*50*/ 0x00000000, /*51*/ 0x00000000, /*52*/ 0x00000000, /*53*/ 0x00000000, /*54*/ 0x00000000, /*55*/ 0x00000000, /*56*/ 0x00000000, /*57*/ 0x00000000, /*58*/ 0x00000000, /*59*/ 0x00000000, /*60*/ 0x00000000, /*61*/ 0x00000000, /*62*/ 0x00000000, /*63*/ 0x00000000, /*64*/ 0x00000000, /*65*/ 0x00000000, /*66*/ 0x00000000, /*67*/ 0x00000000, /*68*/ 0x00000000, /*69*/ 0x00000000, /*70*/ 0x00000000, /*71*/ 0x00000000, /*72*/ 0x00000000, /*73*/ 0x00000000, /*74*/ 0x00000000, /*75*/ 0x00000000, /*76*/ 0x00000000, /*77*/ 0x00000000, /*78*/ 0x00000000, /*79*/ 0x00000000, /*80*/ 0x00000000, /*81*/ 0x00000000, /*82*/ 0x00000000, /*83*/ 0x00000000, /*84*/ 0x00000000 };

// Qprevrmask : Qt[b]: 1-bit = prevr|prevrn, 0-bit otherwise
const uint32_t Qprevrmask [85] = { /*0*/ 0x00000000, /*1*/ 0x00000000, /*2*/ 0x00000000, /*3*/ 0x00000000, /*4*/ 0x00000000, /*5*/ 0x00000000, /*6*/ 0x00000000, /*7*/ 0x00000000, /*8*/ 0x00000000, /*9*/ 0x00000000, /*10*/ 0x00000000, /*11*/ 0x00000000, /*12*/ 0x00000000, /*13*/ 0x00000000, /*14*/ 0x00000000, /*15*/ 0x00000000, /*16*/ 0x00000000, /*17*/ 0x00000000, /*18*/ 0x00000000, /*19*/ 0x00000000, /*20*/ 0x00000000, /*21*/ 0x00000000, /*22*/ 0x00000000, /*23*/ 0x00000000, /*24*/ 0x08000000, /*25*/ 0x00000000, /*26*/ 0x08000000, /*27*/ 0x00000000, /*28*/ 0x08000000, /*29*/ 0x00000000, /*30*/ 0x08000000, /*31*/ 0x00000000, /*32*/ 0x00000000, /*33*/ 0x00000000, /*34*/ 0x00000000, /*35*/ 0x00000000, /*36*/ 0x00000000, /*37*/ 0x00000000, /*38*/ 0x00000000, /*39*/ 0x00000000, /*40*/ 0x00000000, /*41*/ 0x00000000, /*42*/ 0x00000000, /*43*/ 0x00000000, /*44*/ 0x00000000, /*45*/ 0x00000000, /*46*/ 0x00000000, /*47*/ 0x00000000, /*48*/ 0x00000000, /*49*/ 0x00000000, /*50*/ 0x00000000, /*51*/ 0x00000000, /*52*/ 0x00000000, /*53*/ 0x00000000, /*54*/ 0x00000000, /*55*/ 0x00000000, /*56*/ 0x00000000, /*57*/ 0x00000000, /*58*/ 0x00000000, /*59*/ 0x00000000, /*60*/ 0x00000000, /*61*/ 0x00000000, /*62*/ 0x00000000, /*63*/ 0x00000000, /*64*/ 0x00000000, /*65*/ 0x00000000, /*66*/ 0x00000000, /*67*/ 0x00000000, /*68*/ 0x00000000, /*69*/ 0x00000000, /*70*/ 0x00000000, /*71*/ 0x00000000, /*72*/ 0x00000000, /*73*/ 0x00000000, /*74*/ 0x00000000, /*75*/ 0x00000000, /*76*/ 0x00000000, /*77*/ 0x00000000, /*78*/ 0x00000000, /*79*/ 0x00000000, /*80*/ 0x00000000, /*81*/ 0x00000000, /*82*/ 0x00000000, /*83*/ 0x00000000, /*84*/ 0x00000000 };

// Qprev2rmask: Qt[b]: 1-bit = prev2r|prev2rn, 0-bit otherwise
const uint32_t Qprev2rmask[85] = { /*0*/ 0x00000000, /*1*/ 0x00000000, /*2*/ 0x00000000, /*3*/ 0x00000000, /*4*/ 0x00000000, /*5*/ 0x00000000, /*6*/ 0x00000000, /*7*/ 0x00000000, /*8*/ 0x00000000, /*9*/ 0x00000000, /*10*/ 0x00000000, /*11*/ 0x00000000, /*12*/ 0x00000000, /*13*/ 0x00000000, /*14*/ 0x00000000, /*15*/ 0x00000000, /*16*/ 0x00000000, /*17*/ 0x00000000, /*18*/ 0x00000000, /*19*/ 0x00000000, /*20*/ 0x00000000, /*21*/ 0x00000000, /*22*/ 0x00000000, /*23*/ 0x08000000, /*24*/ 0x00000000, /*25*/ 0x00000000, /*26*/ 0x00000000, /*27*/ 0x08000000, /*28*/ 0x00000000, /*29*/ 0x18000000, /*30*/ 0x00000000, /*31*/ 0x00000000, /*32*/ 0x00000000, /*33*/ 0x00000000, /*34*/ 0x00000000, /*35*/ 0x00000000, /*36*/ 0x00000000, /*37*/ 0x00000000, /*38*/ 0x00000000, /*39*/ 0x00000000, /*40*/ 0x00000000, /*41*/ 0x00000000, /*42*/ 0x00000000, /*43*/ 0x00000000, /*44*/ 0x00000000, /*45*/ 0x00000000, /*46*/ 0x00000000, /*47*/ 0x00000000, /*48*/ 0x00000000, /*49*/ 0x00000000, /*50*/ 0x00000000, /*51*/ 0x00000000, /*52*/ 0x00000000, /*53*/ 0x00000000, /*54*/ 0x00000000, /*55*/ 0x00000000, /*56*/ 0x00000000, /*57*/ 0x00000000, /*58*/ 0x00000000, /*59*/ 0x00000000, /*60*/ 0x00000000, /*61*/ 0x00000000, /*62*/ 0x00000000, /*63*/ 0x00000000, /*64*/ 0x00000000, /*65*/ 0x00000000, /*66*/ 0x00000000, /*67*/ 0x00000000, /*68*/ 0x00000000, /*69*/ 0x00000000, /*70*/ 0x00000000, /*71*/ 0x00000000, /*72*/ 0x00000000, /*73*/ 0x00000000, /*74*/ 0x00000000, /*75*/ 0x00000000, /*76*/ 0x00000000, /*77*/ 0x00000000, /*78*/ 0x00000000, /*79*/ 0x00000000, /*80*/ 0x00000000, /*81*/ 0x00000000, /*82*/ 0x00000000, /*83*/ 0x00000000, /*84*/ 0x00000000 };

// Qnextmask  : Qt[b]: 1-bit = next, 0-bit otherwise
const uint32_t Qnextmask  [85] = { /*0*/ 0x00000000, /*1*/ 0x00000000, /*2*/ 0x00000000, /*3*/ 0x00000000, /*4*/ 0x00000000, /*5*/ 0x00000000, /*6*/ 0x00000000, /*7*/ 0x00000000, /*8*/ 0x00000000, /*9*/ 0x00000000, /*10*/ 0x00000000, /*11*/ 0x00000000, /*12*/ 0x00000000, /*13*/ 0x00000000, /*14*/ 0x00000000, /*15*/ 0x00000000, /*16*/ 0x00000000, /*17*/ 0x00000000, /*18*/ 0x00000000, /*19*/ 0x00000000, /*20*/ 0x00000000, /*21*/ 0x00000000, /*22*/ 0x00000000, /*23*/ 0x00000000, /*24*/ 0x00000000, /*25*/ 0x00000000, /*26*/ 0x00000000, /*27*/ 0x00000000, /*28*/ 0x00000000, /*29*/ 0x00000000, /*30*/ 0x00000000, /*31*/ 0x00000000, /*32*/ 0x00000000, /*33*/ 0x00000000, /*34*/ 0x00000000, /*35*/ 0x00000000, /*36*/ 0x00000000, /*37*/ 0x00000000, /*38*/ 0x00000000, /*39*/ 0x00000000, /*40*/ 0x00000000, /*41*/ 0x00000000, /*42*/ 0x00000000, /*43*/ 0x00000000, /*44*/ 0x00000000, /*45*/ 0x00000000, /*46*/ 0x00000000, /*47*/ 0x00000000, /*48*/ 0x00000000, /*49*/ 0x00000000, /*50*/ 0x00000000, /*51*/ 0x00000000, /*52*/ 0x00000000, /*53*/ 0x00000000, /*54*/ 0x00000000, /*55*/ 0x00000000, /*56*/ 0x00000000, /*57*/ 0x00000000, /*58*/ 0x00000000, /*59*/ 0x00000000, /*60*/ 0x00000000, /*61*/ 0x00000000, /*62*/ 0x00000000, /*63*/ 0x00000000, /*64*/ 0x00000000, /*65*/ 0x00000000, /*66*/ 0x00000000, /*67*/ 0x00000000, /*68*/ 0x00000000, /*69*/ 0x00000000, /*70*/ 0x00000000, /*71*/ 0x00000000, /*72*/ 0x00000000, /*73*/ 0x00000000, /*74*/ 0x00000000, /*75*/ 0x00000000, /*76*/ 0x00000000, /*77*/ 0x00000000, /*78*/ 0x00000000, /*79*/ 0x00000000, /*80*/ 0x00000000, /*81*/ 0x00000000, /*82*/ 0x00000000, /*83*/ 0x00000000, /*84*/ 0x00000000 };

// dQ  : additive difference Q'_t - Q_t
const uint32_t dQ         [85] = { /*0*/ 0xfffffffe, /*1*/ 0x00000008, /*2*/ 0x00000000, /*3*/ 0xffffff90, /*4*/ 0xfffff20a, /*5*/ 0x8bfe4122, /*6*/ 0xbfc81884, /*7*/ 0x0100d040, /*8*/ 0xf0000f2f, /*9*/ 0x400000bc, /*10*/ 0xc0000031, /*11*/ 0xd0000021, /*12*/ 0x0fffffe1, /*13*/ 0xbfffffff, /*14*/ 0x00000000, /*15*/ 0x20000000, /*16*/ 0x00000000, /*17*/ 0x80000000, /*18*/ 0x20000000, /*19*/ 0x80000000, /*20*/ 0xc0000000, /*21*/ 0x80000000, /*22*/ 0x60000000, /*23*/ 0x80000000, /*24*/ 0x60000000, /*25*/ 0x00000000, /*26*/ 0xe0000000, /*27*/ 0x80000000, /*28*/ 0x00000000, /*29*/ 0x00000000, /*30*/ 0x00000000, /*31*/ 0x00000000, /*32*/ 0x00000000, /*33*/ 0x00000000, /*34*/ 0x00000000, /*35*/ 0x00000000, /*36*/ 0x00000000, /*37*/ 0x00000000, /*38*/ 0x00000000, /*39*/ 0x00000000, /*40*/ 0x00000000, /*41*/ 0x00000000, /*42*/ 0x00000000, /*43*/ 0x00000000, /*44*/ 0x00000000, /*45*/ 0x00000000, /*46*/ 0x00000000, /*47*/ 0x00000000, /*48*/ 0x00000000, /*49*/ 0x00000000, /*50*/ 0x00000000, /*51*/ 0x00000000, /*52*/ 0x00000000, /*53*/ 0x00000000, /*54*/ 0x00000000, /*55*/ 0x00000000, /*56*/ 0x00000000, /*57*/ 0x00000000, /*58*/ 0x00000000, /*59*/ 0x00000000, /*60*/ 0x00000000, /*61*/ 0x00000000, /*62*/ 0x00000000, /*63*/ 0x00000000, /*64*/ 0x00000000, /*65*/ 0x00000000, /*66*/ 0x00000000, /*67*/ 0x00000000, /*68*/ 0x00000000, /*69*/ 0x00000000, /*70*/ 0x00000000, /*71*/ 0x00000000, /*72*/ 0x00000000, /*73*/ 0x00000000, /*74*/ 0x00000000, /*75*/ 0x00000000, /*76*/ 0x00000000, /*77*/ 0x00000000, /*78*/ 0x00000000, /*79*/ 0x00000000, /*80*/ 0x00000000, /*81*/ 0x00000000, /*82*/ 0x00000000, /*83*/ 0x00000000, /*84*/ 0x00000000 };


namespace basesolgen {
#include "tables_org.hpp"     // org_msgbitrels{,80}{,_size}
#include "tables_wboom.hpp"   // boom_msgbitrels{,80}{,_size}
#include "tables_wQ17nb.hpp"  // Q17nb_msgbitrels{,80}{,_size}
#include "tables_wQ14nb.hpp"  // Q14nb_msgbitrels{,80}{,_size}
}
using namespace basesolgen;

// disturbance vector tables
const uint32_t DV_DV[80] = { /*0*/ 0x80000000, /*1*/ 0xe0000000, /*2*/ 0x20000000, /*3*/ 0xd0000000, /*4*/ 0x80000000, /*5*/ 0xe0000000, /*6*/ 0xa0000000, /*7*/ 0x10000000, /*8*/ 0x80000000, /*9*/ 0xe0000000, /*10*/ 0x20000000, /*11*/ 0xc0000000, /*12*/ 0x80000000, /*13*/ 0x60000000, /*14*/ 0x80000000, /*15*/ 0xc0000000, /*16*/ 0x80000000, /*17*/ 0xa0000000, /*18*/ 0x80000000, /*19*/ 0xe0000000, /*20*/ 0x00000000, /*21*/ 0x20000000, /*22*/ 0x80000000, /*23*/ 0x60000000, /*24*/ 0x00000000, /*25*/ 0x00000000, /*26*/ 0x80000000, /*27*/ 0x80000000, /*28*/ 0x00000000, /*29*/ 0x00000000, /*30*/ 0x00000000, /*31*/ 0x00000000, /*32*/ 0x00000000, /*33*/ 0x80000000, /*34*/ 0x00000000, /*35*/ 0x80000000, /*36*/ 0x00000000, /*37*/ 0x80000000, /*38*/ 0x00000000, /*39*/ 0xc0000000, /*40*/ 0x00000000, /*41*/ 0x00000000, /*42*/ 0x80000000, /*43*/ 0x00000000, /*44*/ 0x00000000, /*45*/ 0x00000000, /*46*/ 0x00000000, /*47*/ 0x80000000, /*48*/ 0x00000000, /*49*/ 0x00000000, /*50*/ 0x00000000, /*51*/ 0x00000000, /*52*/ 0x00000000, /*53*/ 0x80000000, /*54*/ 0x00000000, /*55*/ 0x80000000, /*56*/ 0x00000000, /*57*/ 0x00000000, /*58*/ 0x00000000, /*59*/ 0x00000000, /*60*/ 0x00000000, /*61*/ 0x00000000, /*62*/ 0x00000000, /*63*/ 0x00000000, /*64*/ 0x00000000, /*65*/ 0x00000000, /*66*/ 0x00000000, /*67*/ 0x00000001, /*68*/ 0x00000000, /*69*/ 0x00000000, /*70*/ 0x00000002, /*71*/ 0x00000001, /*72*/ 0x00000000, /*73*/ 0x00000004, /*74*/ 0x00000002, /*75*/ 0x00000002, /*76*/ 0x00000008, /*77*/ 0x00000004, /*78*/ 0x00000000, /*79*/ 0x00000012 };
const uint32_t DV_DW[80] = { /*0*/ 0x0c000002, /*1*/ 0xc0000010, /*2*/ 0xb400001c, /*3*/ 0x3c000004, /*4*/ 0xbc00001a, /*5*/ 0x20000010, /*6*/ 0x2400001c, /*7*/ 0xec000014, /*8*/ 0x0c000002, /*9*/ 0xc0000010, /*10*/ 0xb400001c, /*11*/ 0x2c000004, /*12*/ 0xbc000018, /*13*/ 0xb0000010, /*14*/ 0x0000000c, /*15*/ 0xb8000010, /*16*/ 0x08000018, /*17*/ 0x78000010, /*18*/ 0x08000014, /*19*/ 0x70000010, /*20*/ 0xb800001c, /*21*/ 0xe8000000, /*22*/ 0xb0000004, /*23*/ 0x58000010, /*24*/ 0xb000000c, /*25*/ 0x48000000, /*26*/ 0xb0000000, /*27*/ 0xb8000010, /*28*/ 0x98000010, /*29*/ 0xa0000000, /*30*/ 0x00000000, /*31*/ 0x00000000, /*32*/ 0x20000000, /*33*/ 0x80000000, /*34*/ 0x00000010, /*35*/ 0x00000000, /*36*/ 0x20000010, /*37*/ 0x20000000, /*38*/ 0x00000010, /*39*/ 0x60000000, /*40*/ 0x00000018, /*41*/ 0xe0000000, /*42*/ 0x90000000, /*43*/ 0x30000010, /*44*/ 0xb0000000, /*45*/ 0x20000000, /*46*/ 0x20000000, /*47*/ 0xa0000000, /*48*/ 0x00000010, /*49*/ 0x80000000, /*50*/ 0x20000000, /*51*/ 0x20000000, /*52*/ 0x20000000, /*53*/ 0x80000000, /*54*/ 0x00000010, /*55*/ 0x00000000, /*56*/ 0x20000010, /*57*/ 0xa0000000, /*58*/ 0x00000000, /*59*/ 0x20000000, /*60*/ 0x20000000, /*61*/ 0x00000000, /*62*/ 0x00000000, /*63*/ 0x00000000, /*64*/ 0x00000000, /*65*/ 0x00000000, /*66*/ 0x00000000, /*67*/ 0x00000001, /*68*/ 0x00000020, /*69*/ 0x00000001, /*70*/ 0x40000002, /*71*/ 0x40000041, /*72*/ 0x40000022, /*73*/ 0x80000005, /*74*/ 0xc0000082, /*75*/ 0xc0000046, /*76*/ 0x4000004b, /*77*/ 0x80000107, /*78*/ 0x00000089, /*79*/ 0x00000014 };



// [-50] global tables
// M and M' message and state tables
uint32_t main_m1[80];
uint32_t main_m2[80];
uint32_t main_Q1[85];
uint32_t main_Q2[85];
uint32_t main_Q1r30[85];
uint32_t main_Q2r30[85];

// msgbitrel precomputation word[i][j] bit b
// = precomputed bitrelation for W_mainblockoffset+j bit b for step mainblockoffset+i
// = sum of Wk[l] bits (where k was done from start up to (not incl) step mainblockoffset+i) and the constant (0 or 1)
uint32_t msgbitrel_precomp[16][16];

// [-40] helper functions
inline unsigned hamming_weight(uint32_t w) { return __builtin_popcount(w); }

// [-1] 

// [1000] verify stateconditions and messagebitrelations on an step interval
size_t verifycount = 0;
//bool verify(int firststep, int laststep, int lastQ = 80, uint32* mQ1 = main_Q1, uint32* mm1 = main_m1, mbrset_t mbrset = MBR_Q14NB )
bool verify(int firststep, int laststep, int lastQ, const uint32_t* mQ1, const uint32_t* mm1, mbrset_t mbrset)
{
/*
	// [1010] output call count
	if (hamming_weight(++verifycount) == 1)
	{
		std::cout << "(" << verifycount << ")" << std::flush;
	}
*/
	// [1100] verify stateconditions
	bool ok = true;
	for (int t = firststep - 4; t <= laststep+1 && t <= lastQ; ++t)
	{
		if (0 != (Qcondmask[Qoffset+t] & (
			  mQ1[Qoffset+t]
			^ Qset1mask[Qoffset+t]
			^ ( Qprevmask[Qoffset+t]   & mQ1[Qoffset+t-1] )
			^ ( Qprevrmask[Qoffset+t]  & rotate_left(mQ1[Qoffset+t-1],30) )
			^ ( Qprev2rmask[Qoffset+t] & rotate_left(mQ1[Qoffset+t-2],30) )
			^ ( Qnextmask[Qoffset+t]   & mQ1[Qoffset+t+1] )
			) ))
		{
#ifndef SILENTERROR
			std::cerr << "Q_" << t << " does not satisfy conditions!" << std::endl;
#endif
			ok = false; //return false;
		}
	}
	
	// [1200] verify message bitrelations
	unsigned msgbitrels_size;
	switch (mbrset) {
	  case MBR_ORG: msgbitrels_size = tbl_org::msgbitrels_size; break;
	  case MBR_BOOM: msgbitrels_size = tbl_boom::msgbitrels_size; break;
	  case MBR_Q17NB: msgbitrels_size = tbl_nbQ17::msgbitrels_size; break;
	  case MBR_Q14NB: msgbitrels_size = tbl_nbQ14::msgbitrels_size; break;
	}
	for (unsigned r = 0; r < msgbitrels_size; ++r)
	{
		bool okay = true;
		const uint32_t* mbrr;
                switch (mbrset) {
		  case MBR_ORG:   mbrr = tbl_org::msgbitrels[r]; break;
		  case MBR_BOOM:  mbrr = tbl_boom::msgbitrels[r]; break;
		  case MBR_Q17NB: mbrr = tbl_nbQ17::msgbitrels[r]; break;
		  case MBR_Q14NB: mbrr = tbl_nbQ14::msgbitrels[r]; break;
		}
		uint32_t w = mbrr[16];
		for (unsigned t = mainblockoffset; t < mainblockoffset+16; ++t)
		{
		        uint32_t mbrmask = mbrr[t-mainblockoffset];
			if ((t < firststep | t > laststep) && mbrmask!=0)
			{
				okay = false;
				break;
			}
			w ^= mm1[t] & mbrmask;
		}
		if (okay && 0 != (hamming_weight(w)&1) )
		{
#ifndef SILENTERROR
			std::cerr << "bitrelation " << r << " is not satisfied!" << std::endl;
#endif
                        ok = false;
//			return false;
		}
	}
	
	// [1300] verify step computations
	for (int t = firststep; t <= laststep; ++t)
	{
		uint32_t f;
		if (t >=  0 && t<20) f = sha1_f1(mQ1[Qoffset+t-1],rotate_left(mQ1[Qoffset+t-2],30),rotate_left(mQ1[Qoffset+t-3],30));
		if (t >= 20 && t<40) f = sha1_f2(mQ1[Qoffset+t-1],rotate_left(mQ1[Qoffset+t-2],30),rotate_left(mQ1[Qoffset+t-3],30));
		if (t >= 40 && t<60) f = sha1_f3(mQ1[Qoffset+t-1],rotate_left(mQ1[Qoffset+t-2],30),rotate_left(mQ1[Qoffset+t-3],30));
		if (t >= 60 && t<80) f = sha1_f4(mQ1[Qoffset+t-1],rotate_left(mQ1[Qoffset+t-2],30),rotate_left(mQ1[Qoffset+t-3],30));
		uint32_t Qtp1 = rotate_left(mQ1[Qoffset+t],5) + f + rotate_left(mQ1[Qoffset+t-4],30) + mm1[t] + sha1_ac[t/20];
		if (Qtp1 != mQ1[Qoffset+t+1])
		{
#ifndef SILENTERROR
			std::cerr << "step " << t << " is incorrect!" << std::endl;
#endif
			ok=false;//return false;
		}
	}
	return ok;//true;
// [1900] end verify()
}



void step14nb(const uint32_t* mm1, const uint32_t* mQ1);

/*extern*/ std::vector<q14sol_t> q14sols;
void process_q14sol(const uint32_t mm1[80], const uint32_t mQ1[85])
{
    if (!verify(0,15,14, mQ1, mm1, MBR_Q17NB))
    {
#ifndef SILENTERROR
        std::cerr << "process_q14sol(): not a Q14solution" << std::endl;
        exit(1);
#endif //SILENTERROR
    }
    else
    {
        q14sols.push_back( q14sol_t() );
        memcpy(q14sols.back().m, mm1, 16*4);
        memcpy(q14sols.back().Q, &mQ1[5], 16*4);

//        verify(q14sols.back());        
//        step14nb(mm1, mQ1);
    }
}


/*extern*/ int max_q13sols;
/*extern*/ std::vector<q13sol_t> q13sols;
void process_q13sol(const uint32_t mm1[80], const uint32_t mQ1[85])
{
    if (!verify(0,15,13, mQ1, mm1, MBR_Q14NB))
    {
#ifndef SILENTERROR
        std::cerr << "process_q13sol(): not a Q13solution" << std::endl;
        exit(1);
#endif //SILENTERROR
    }
    else
    {
        q13sols.push_back( q13sol_t() );
        memcpy(q13sols.back().m, mm1, 16*4);
      
        step13nb(mm1, mQ1);
      
        if (q14sols.size() > MINQ14SOLPERJOB)
        {
            
            std::cout << "Writing " << q13sols.size() << " Q13sols (with " << q14sols.size() << " #Q14sols) to " << outputfile << "..." << std::flush;
            
            std::ofstream ofs(outputfile.c_str(), std::ios::app);
            if (!ofs)
            {
                std::cerr << "Cannot open basesol.txt!" << std::endl;
                exit(1);  
            }
            for (size_t i = 0; i < q13sols.size(); ++i)
                ofs << encode_q13sol( q13sols[i] ) << " ";
            ofs << std::endl;

            q14sols.clear();
            q13sols.clear();
            std::cout << "done." << std::endl;
            
            static size_t q13solcnt = 0;
            if (max_basesols > 0 && ++q13solcnt == max_basesols)
              exit(0);
        }        
    }
}
















void step14nb(const uint32_t* mm1, const uint32_t* mQ1)
{
/* neutral bits
W10[6]fwQ13 0.268555 fwQ14 1.00098 fwQ15 52.2949 Q15[0]2.14844 Q15[1]1.12305 Q15[31]6.5918 Q16[29]4.49219 Q16[30]4.02832
W10[7]fwQ13 0.341797 fwQ14 1.78223 fwQ15 99.5605 Q15[0]4.02832 Q15[1]2.05078 Q16[29]7.2998 Q16[30]6.5918 Q16[31]6.68945
W10[8]fwQ13 0.683594 fwQ14 3.125 fwQ15 55.542 Q15[0]7.54395 Q15[1]4.07715 Q15[27]1.75781
W10[9]fwQ12 0.0244141 fwQ13 1.46484 fwQ14 6.03027 fwQ15 99.9023 Q15[1]7.61719 Q15[27]2.73438 Q16[1]9.05762
W10[10]fwQ13 2.14844 fwQ14 9.66797 fwQ15 64.5996 Q15[27]5.07812 Q15[29]1.78223
W11[9]fwQ13 0.0488281 fwQ14 0.390625 fwQ15 13.5498 Q15[0]0.78125 Q15[1]0.341797 Q15[29]2.92969 Q15[31]1.48926 Q16[1]6.86035 Q16[27]3.83301 Q17[1]8.3252 Q17[31]7.9834
W11[10] w14[11]fwQ13 0.12207 fwQ14 0.610352 fwQ15 24.2432 Q15[0]1.26953 Q15[1]0.585938 Q15[29]5.46875 Q15[31]2.58789 Q16[27]5.27344 Q16[29]2.05078
W11[11]fwQ13 0.146484 fwQ14 1.02539 fwQ15 50.4639 Q15[0]2.24609 Q15[1]1.07422 Q15[31]5.37109 Q16[29]3.78418 Q16[30]3.58887
W11[12] w14[11]fwQ13 0.268555 fwQ14 1.70898 fwQ15 99.8047 Q15[0]4.12598 Q15[1]1.92871 Q16[29]5.9082 Q16[30]5.32227 Q16[31]5.37109
W11[13] w13[14] w13[20] w13[21] w13[24] w14[11] w14[13] w14[16] w14[18] w14[20] w14[21] w15[14] w15[15] w15[16] w15[17] w15[30]fwQ13 0.634766 fwQ14 6.17676 fwQ15 83.5205 Q15[1]6.95801
W11[14] w15[11]fwQ13 1.00098 fwQ14 5.46875 fwQ15 99.9756 Q15[1]7.15332 Q15[27]2.0752 Q16[1]7.9834 Q17[28]9.86328
W11[15] w13[20] w13[21] w13[24] w14[11] w14[13] w14[16] w14[18] w14[20] w14[21] w15[14] w15[15] w15[16] w15[17] w15[30]fwQ13 1.97754 fwQ14 9.61914 fwQ15 99.9268
W12[2] w13[20] w13[31] w14[1] w14[17] w14[18] w14[19] w14[20] w14[21] w14[22] w14[23] w14[30] w15[2] w15[15] w15[17] w15[18] w15[19] w15[20] w15[21] w15[22] w15[24]fwQ14 0.756836 fwQ15 100 Q15[0]1.34277 Q15[29]6.0791
W12[14] w14[11] w14[13]fwQ13 0.0244141 fwQ14 0.244141 fwQ15 13.2568 Q15[0]0.708008 Q15[1]0.268555 Q15[29]2.9541 Q15[31]1.58691 Q16[1]6.44531 Q16[27]2.56348 Q17[1]7.4707 Q17[31]6.29883
W12[15] w14[12] w14[14] w15[11]fwQ13 0.170898 fwQ14 0.756836 fwQ15 25.8301 Q15[0]1.48926 Q15[1]0.488281 Q15[29]6.32324 Q15[31]2.97852 Q16[27]4.24805 Q16[29]1.75781
W12[16] w13[14] w13[20] w13[21] w13[24] w14[11] w14[15] w14[16] w14[18] w14[20] w14[21] w15[11] w15[14] w15[15] w15[16] w15[17] w15[30]fwQ13 0.219727 fwQ14 6.66504 fwQ15 98.2178 Q15[1]7.73926
W12[17] w14[14] w14[16] w15[13]fwQ13 0.390625 fwQ14 1.78223 fwQ15 99.8047 Q15[0]4.05273 Q15[1]1.95312 Q16[29]4.66309 Q16[30]4.41895 Q16[31]4.54102
W12[18] w13[14] w14[12] w14[15] w14[17] w15[11] w15[13] w15[14]fwQ13 0.830078 fwQ14 3.51562 fwQ15 55.9082 Q15[0]7.8125 Q15[1]3.83301 Q15[27]1.07422 Q16[29]9.52148 Q16[30]8.81348 Q16[31]9.05762
W12[19] w14[11] w14[12] w14[13] w14[16] w14[18] w15[11] w15[14] w15[15]fwQ13 1.19629 fwQ14 5.78613 fwQ15 100 Q15[1]7.27539 Q15[27]1.3916 Q16[1]7.83691
W12[20] w13[14] w13[19] w13[21] w13[24] w14[11] w14[13] w14[14] w14[16] w14[18] w14[19] w14[21] w15[2] w15[11] w15[13] w15[14] w15[15] w15[20] w15[21] w15[24] w15[30]fwQ13 2.44141 fwQ14 9.74121 fwQ15 100
W13[12] w13[14] w13[20] w13[21] w13[24] w14[11] w14[13] w14[16] w14[18] w14[20] w14[21] w15[14] w15[15] w15[16] w15[17] w15[30]fwQ14 6.64062 fwQ15 100 Q15[1]7.76367
W13[16] w13[19] w13[20] w14[17] w14[20] w15[2] w15[11] w15[15] w15[17] w15[20] w15[21] w15[24]fwQ14 0.732422 fwQ15 26.3184 Q15[0]1.34277 Q15[1]0.634766 Q15[29]5.9082 Q15[31]2.90527
*/
      static uint64_t Q14okcnt = 0;
      static uint64_t Q15okcnt = 0;
      ++Q14okcnt;

	static const int t = 14;

	uint32_t m1[80];
	uint32_t Q1[85];
	
	memcpy(m1,mm1,16*4);
	memcpy(Q1,mQ1,21*4);


  const uint32_t Q14bu = mQ1[Qoffset+14];

  for (int W10t6b = 0; W10t6b < 2; ++W10t6b) { m1[10] ^= 0x40;  
  for (int W10t7b = 0; W10t7b < 2; ++W10t7b) { m1[10] ^= 0x80;  
  for (int W10t8b = 0; W10t8b < 2; ++W10t8b) { m1[10] ^= 0x100;  
  for (int W10t9b = 0; W10t9b < 2; ++W10t9b) { m1[10] ^= 0x200;  
  for (int W10t10b = 0; W10t10b < 2; ++W10t10b) { m1[10] ^= 0x400; 
      sha1_step<10>(Q1,m1);
      if ((Q1[Qoffset+10+1]^mQ1[Qoffset+10+1])&Qcondmask[Qoffset+10+1])
        continue;

  for (int W11t13b = 0; W11t13b < 2; ++W11t13b) { m1[11] ^= 0x2000; m1[13] ^= 0x1304000; m1[14] ^= 0x352800; m1[15] ^= 0x4003c000;    
  for (int W11t15b = 0; W11t15b < 2; ++W11t15b) { m1[11] ^= 0x8000; m1[13] ^= 0x1300000; m1[14] ^= 0x352800; m1[15] ^= 0x4003c000;  
  for (int W11t10b = 0; W11t10b < 2; ++W11t10b) { m1[11] ^= 0x400; m1[14] ^= 0x800;  
  for (int W11t12b = 0; W11t12b < 2; ++W11t12b) { m1[11] ^= 0x1000; m1[14] ^= 0x800;  
  for (int W11t14b = 0; W11t14b < 2; ++W11t14b) { m1[11] ^= 0x4000; m1[15] ^= 0x800;  
  for (int W11t11b = 0; W11t11b < 2; ++W11t11b) { m1[11] ^= 0x800; 
  for (int W11t9b = 0; W11t9b < 2; ++W11t9b) { m1[11] ^= 0x200;  
      sha1_step<11>(Q1,m1);
      if ((Q1[Qoffset+11+1]^mQ1[Qoffset+11+1])&Qcondmask[Qoffset+11+1])
        continue;

  for (int W12t2b = 0; W12t2b < 2; ++W12t2b) { m1[12] ^= 0x4; m1[13] ^= 0x80100000; m1[14] ^= 0x40fe0002; m1[15] ^= 0x17e8004;  
  for (int W12t16b = 0; W12t16b < 2; ++W12t16b) { m1[12] ^= 0x10000; m1[13] ^= 0x1304000; m1[14] ^= 0x358800; m1[15] ^= 0x4003c800;  
  for (int W12t18b = 0; W12t18b < 2; ++W12t18b) { m1[12] ^= 0x40000; m1[13] ^= 0x4000; m1[14] ^= 0x29000; m1[15] ^= 0x6800;  
  for (int W12t20b = 0; W12t20b < 2; ++W12t20b) { m1[12] ^= 0x100000; m1[13] ^= 0x1284000; m1[14] ^= 0x2d6800; m1[15] ^= 0x4130e804;  
  for (int W12t15b = 0; W12t15b < 2; ++W12t15b) { m1[12] ^= 0x8000; m1[14] ^= 0x5000; m1[15] ^= 0x800;  
  for (int W12t17b = 0; W12t17b < 2; ++W12t17b) { m1[12] ^= 0x20000; m1[14] ^= 0x14000; m1[15] ^= 0x2000;  
  for (int W12t19b = 0; W12t19b < 2; ++W12t19b) { m1[12] ^= 0x80000; m1[14] ^= 0x53800; m1[15] ^= 0xc800;  
  for (int W12t14b = 0; W12t14b < 2; ++W12t14b) { m1[12] ^= 0x4000; m1[14] ^= 0x2800;  
      sha1_step<12>(Q1,m1);
      if ((Q1[Qoffset+12+1]^mQ1[Qoffset+12+1])&Qcondmask[Qoffset+12+1])
        continue;

  for (int W13t12b = 0; W13t12b < 2; ++W13t12b) { m1[13] ^= 0x1305000; m1[14] ^= 0x352800; m1[15] ^= 0x4003c000;  
  for (int W13t16b = 0; W13t16b < 2; ++W13t16b) { m1[13] ^= 0x190000; m1[14] ^= 0x120000; m1[15] ^= 0x1328804;  
      sha1_step<13>(Q1,m1);
      if ((Q1[Qoffset+13+1]^Q14bu)&Qcondmask[Qoffset+13+1])
        continue;

      sha1_step<14>(Q1,m1);

      if ( 
          ( Q1[Qoffset+15]
            ^ Qset1mask[Qoffset+15]
            ^ (Qprevmask[Qoffset+15]&Q1[Qoffset+14])
          ) & Qcondmask[Qoffset+15]
          )
            continue;


      sha1_step<15>(Q1,m1);

      if (hamming_weight(++Q15okcnt)==1)
        std::cout << "(Q14ok:" << Q14okcnt << " Q15ok:" << Q15okcnt << " "<< log(double(Q15okcnt)/double(Q14okcnt))/log(2.0) << ")" << std::flush;              
      verify(0,15,15,Q1,m1,MBR_Q17NB);

//  }
  }}       // 2 W13nb
  }}}}}}}} // 8 W12nb
  }}}}}}}  // 7 W11nb
  }}}}}    // 5 W10nb
  
}


void step13nb(const uint32_t* mm1, const uint32_t* mQ1)
{
/* neutral bits
W10[11] w15[11]fwQ12 0.0732422 fwQ13 3.68652 fwQ14 16.5283 Q14[1]1.7334 Q15[27]9.10645 Q15[29]3.58887
W11[16] w15[13]fwQ12 0.0976562 fwQ13 3.80859 fwQ14 17.041 Q14[1]1.92871 Q15[27]7.10449 Q15[29]2.68555
W11[17] w13[14] w13[19] w13[20] w14[11] w14[13] w14[17] w14[20] w15[2] w15[11] w15[14] w15[15] w15[17] w15[20] w15[21] w15[24]fwQ12 0.195312 fwQ13 6.64062 fwQ14 31.0791 Q14[1]2.83203 Q15[29]8.37402 Q15[31]4.32129
W12[0] w13[14] w13[17] w13[24] w14[11] w14[14] w14[16] w15[11] w15[16] w15[18] w15[21]fwQ14 53.5645 Q14[29]6.39648 Q15[27]3.73535
W12[21] w13[17] w14[15] w14[18] w14[20] w15[13] w15[14] w15[16] w15[17]fwQ13 4.39453 fwQ14 17.2119 Q14[1]2.05078 Q15[27]6.39648 Q15[29]2.44141
W12[22] w13[18] w13[20] w13[21] w13[24] w14[11] w14[12] w14[13] w14[18] w14[19] w14[20] w15[16] w15[18] w15[30]fwQ13 8.3252 fwQ14 31.9824 Q14[1]3.6377
W13[0] w13[17] w13[19] w13[20] w13[21] w14[12] w14[13] w14[14] w14[16] w14[17] w14[18] w14[20] w15[2] w15[13] w15[14] w15[15] w15[17] w15[18] w15[24]fwQ14 50.2441 Q14[29]1.17188 Q15[31]6.29883
W13[1] w13[14] w13[17] w13[18] w13[19] w13[20] w13[24] w14[12] w14[13] w14[15] w14[16] w14[20] w14[22] w15[2] w15[11] w15[13] w15[15] w15[17] w15[19] w15[20] w15[21] w15[23] w15[30]fwQ14 100 Q14[29]6.37207
*/

	static const int t = 13;

	uint32_t m1[80];
	uint32_t Q1[85];
	
	memcpy(m1,mm1,16*4);
	memcpy(Q1,mQ1,21*4);

	unsigned okcnt = 0;
	
	for (int W10t11b = 0; W10t11b < 2; ++W10t11b) { m1[10] ^= 0x800; m1[15] ^= 0x800;  
            sha1_step<10>(Q1,m1);
            if ((Q1[Qoffset+10+1]^mQ1[Qoffset+10+1])&Qcondmask[Qoffset+10+1])
                continue;

	for (int W11t16b = 0; W11t16b < 2; ++W11t16b) { m1[11] ^= 0x10000; m1[15] ^= 0x2000;  
	for (int W11t17b = 0; W11t17b < 2; ++W11t17b) { m1[11] ^= 0x20000; m1[13] ^= 0x184000; m1[14] ^= 0x122800; m1[15] ^= 0x132c804;  
            sha1_step<11>(Q1,m1);
            if ((Q1[Qoffset+11+1]^mQ1[Qoffset+11+1])&Qcondmask[Qoffset+11+1])
                continue;

	for (int W12t0b = 0; W12t0b < 2; ++W12t0b) { m1[12] ^= 0x1; m1[13] ^= 0x1024000; m1[14] ^= 0x14800; m1[15] ^= 0x250800;  
	for (int W12t21b = 0; W12t21b < 2; ++W12t21b) { m1[12] ^= 0x200000; m1[13] ^= 0x20000; m1[14] ^= 0x148000; m1[15] ^= 0x36000;  
	for (int W12t22b = 0; W12t22b < 2; ++W12t22b) { m1[12] ^= 0x400000; m1[13] ^= 0x1340000; m1[14] ^= 0x1c3800; m1[15] ^= 0x40050000;
            sha1_step<12>(Q1,m1);
            if ((Q1[Qoffset+12+1]^mQ1[Qoffset+12+1])&Qcondmask[Qoffset+12+1])
                continue;

	for (int W13t0b = 0; W13t0b < 2; ++W13t0b) { m1[13] ^= 0x3a0001; m1[14] ^= 0x177000; m1[15] ^= 0x106e004; 
	for (int W13t1b = 0; W13t1b < 2; ++W13t1b) { m1[13] ^= 0x11e4002; m1[14] ^= 0x51b000; m1[15] ^= 0x40baa804; 
            sha1_step<13>(Q1,m1);
            if ( 
              ( Q1[Qoffset+14]
                ^ Qset1mask[Qoffset+14]
                ^ (Qprevmask[Qoffset+14]&Q1[Qoffset+13])
              ) & Qcondmask[Qoffset+14]
              )
                continue;

            sha1_step<14>(Q1,m1);
            sha1_step<15>(Q1,m1);
            
            process_q14sol(m1,Q1);
		
	}}  // 2 W13nb
	}}} // 3 W12nb
	}}  // 2 W11nb
	}   // 1 W10nb


/*	
	++nb13cnt[okcnt];
	
	static uint64_t x = 0;
	
	if (hamming_weight(++x) == 1)
	{
	  double avg = 0, cnt = 0;
	  for (auto cc : nb13cnt)
	  {
	    std::cout << "nb13cnt: " << cc.first << " => " << cc.second << std::endl;
	    avg += cc.first * cc.second;
	    cnt += cc.second;
          }
          std::cout << "avg=" << avg/cnt << std::endl;
	}
*/
}



// [99997] 
uint32_t successcount15 = 0;
// [100000] step15 forward
void step15()
{
	// [100001] define step number as constant
	static const int t = 15;
	// [100002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [100003] 
	if (t != mainblockoffset+15) successcount15 = 0;
	// [100040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0xe8000002; // 11101000000000000000000000000010
	const uint32_t mvalmask = 0xffffffff; // 11111111111111111111111111111111
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t loopmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t mval2 = mval | (~Qtp1val & fixmask) | (rng() & loopmask);
	const uint32_t Qtp1valmask2 = Qtp1valmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [100100] start outer main loop
	static const uint32_t freebitsval_outer = 0; 
	{
		// [100150] start inner main loop
		static const uint32_t freebitsval_inner = 0; 
		{
			// [100200] compute value
			uint32_t m = (mval2^freebitsval_inner^freebitsval_outer);
			uint32_t Qtp1 = precomp + m;
			//if ( (Qtp1^Qtp1val) & Qtp1valmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			static const uint32_t fix = 0; //(Qtp1^Qtp1val) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [100400] update msgbitrels
			
			// [100600] verify
			process_q13sol(main_m1, main_Q1);
//			verify(0,15,13,main_Q1,main_m1,MBR_Q14NB); // verify all msgbitrel, but stateconds up to Q13
			//continue;
			// [100699] 
			//if (++successcount15 == (1<<2) && 0 == successcount15) return;
		// [100800] end inner main loop
		};
	// [100850] end outer main loop
	};
// [100900] 
}


// [109997] 
uint32_t successcount14 = 0;
// [110000] step14 forward
void step14()
{
	// [110001] define step number as constant
	static const int t = 14;
	// [110002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [110003] 
	if (t != mainblockoffset+15) successcount14 = 0;
	// [110040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0xa8000003; // 10101000000000000000000000000011
	const uint32_t mvalmask = 0xffffffff; // 11111111111111111111111111111111
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t loopmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t mval2 = mval | (~Qtp1val & fixmask) | (rng() & loopmask);
	const uint32_t Qtp1valmask2 = Qtp1valmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [110100] start outer main loop
	static const uint32_t freebitsval_outer = 0; 
	{
		// [110150] start inner main loop
		static const uint32_t freebitsval_inner = 0; 
		{
			// [110200] compute value
			uint32_t m = (mval2^freebitsval_inner^freebitsval_outer);
			uint32_t Qtp1 = precomp + m;
			// if ( (Qtp1^Qtp1val) & Qtp1valmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			static const uint32_t fix = 0; // (Qtp1^Qtp1val) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [110400] update msgbitrels
			msgbitrel_precomp[15 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			
			// [110699] 
			//if (++successcount14 == (1<<2) && 0 == successcount15) return;
			// [110700] call next step
			step15();
		// [110800] end inner main loop
		};
	// [110850] end outer main loop
	};
// [110900] 
}


// [119997] 
uint32_t successcount13 = 0;
// [120000] step13 forward
void step13()
{
	// [120001] define step number as constant
	static const int t = 13;
	// [120002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [120003] 
	if (t != mainblockoffset+15) successcount13 = 0;
	// [120040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0x20000002; // 00100000000000000000000000000010
	const uint32_t mvalmask = 0xffffffff; // 11111111111111111111111111111111
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t loopmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t mval2 = mval | (~Qtp1val & fixmask) | (rng() & loopmask);
	const uint32_t Qtp1valmask2 = Qtp1valmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [120100] start outer main loop
	static const uint32_t freebitsval_outer = 0; 
	{
		// [120150] start inner main loop
		static const uint32_t freebitsval_inner = 0; 
		{
			// [120200] compute value
			uint32_t m = (mval2^freebitsval_inner^freebitsval_outer);
			uint32_t Qtp1 = precomp + m;
			// if ( (Qtp1^Qtp1val) & Qtp1valmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			static const uint32_t fix = 0; // (Qtp1^Qtp1val) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [120400] update msgbitrels
			msgbitrel_precomp[14 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[14 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			
			// [120699] 
			//if (++successcount13 == (1<<2) && 0 == successcount15) return;
			// [120700] call next step
			step14();
		// [120800] end inner main loop
		};
	// [120850] end outer main loop
	};
// [120900] 
}


// [129997] 
uint32_t successcount12 = 0;
// [130000] step12 forward
void step12()
{
	// [130001] define step number as constant
	static const int t = 12;
	// [130002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [130003] 
	if (t != mainblockoffset+15) successcount12 = 0;
	// [130040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0x88000000; // 10001000000000000000000000000000
	const uint32_t mvalmask = 0x3c7ffffd; // 00111100011111111111111111111101
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x80000000; // 10000000000000000000000000000000
	const uint32_t loopmask = 0x43800002; // 01000011100000000000000000000010
	const uint32_t mval2 = mval | (~Qtp1val & fixmask) | (rng() & loopmask);
	const uint32_t Qtp1valmask2 = Qtp1valmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [130100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x03800000; // 00000011100000000000000000000000
		// [130150] start inner main loop
		uint32_t freebitsval_inner = 0;
		do
		{
			--freebitsval_inner; freebitsval_inner &= 0x40000002; // 01000000000000000000000000000010
			// [130200] compute value
			uint32_t m = (mval2^freebitsval_inner^freebitsval_outer);
			uint32_t Qtp1 = precomp + m;
			if ( (Qtp1^Qtp1val) & Qtp1valmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (Qtp1^Qtp1val) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [130210] filter inter-word dependencies
			if (0 != (1 & hamming_weight( (main_m1[t] & 0x80000002) ^ (mval & (1<<31) ) ) ) ) continue; // 10000000000000000000000000000010
			// [130400] update msgbitrels
			msgbitrel_precomp[13 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[13 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000002)&1)<<14;
			msgbitrel_precomp[13 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<17;
			msgbitrel_precomp[13 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<18;
			msgbitrel_precomp[13 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41800000)&1)<<19;
			msgbitrel_precomp[13 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41000002)&1)<<20;
			msgbitrel_precomp[13 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03000002)&1)<<21;
			msgbitrel_precomp[13 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41000002)&1)<<24;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<0;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000000)&1)<<11;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02800000)&1)<<12;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41800002)&1)<<13;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000000)&1)<<14;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x42000002)&1)<<15;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41000002)&1)<<16;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01800000)&1)<<17;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000002)&1)<<19;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00800002)&1)<<20;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000002)&1)<<21;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x42800002)&1)<<22;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000000)&1)<<23;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<24;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000000)&1)<<30;
			msgbitrel_precomp[13 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000000)&1)<<31;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000000)&1)<<2;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000002)&1)<<11;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000000)&1)<<13;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41000002)&1)<<14;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40800002)&1)<<15;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40800002)&1)<<16;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x43000000)&1)<<17;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02800002)&1)<<18;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41800000)&1)<<19;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x42000002)&1)<<20;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03000000)&1)<<21;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<22;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<23;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41000000)&1)<<24;
			msgbitrel_precomp[13 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000000)&1)<<30;
			
			// [130699] 
			if (++successcount12 == (1<<2) && 0 == successcount15) return;
			// [130700] call next step
			step13();
		// [130800] end inner main loop
		} while (freebitsval_inner != 0);
	// [130850] end outer main loop
	} while (freebitsval_outer != 0);
// [130900] 
}


// [139997] 
uint32_t successcount11 = 0;
// [140000] step11 forward
void step11()
{
	// [140001] define step number as constant
	static const int t = 11;
	// [140002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [140003] 
	if (t != mainblockoffset+15) successcount11 = 0;
	// [140040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0x88000022; // 10001000000000000000000000100010
	const uint32_t mvalmask = 0x3c03ffdc; // 00111100000000111111111111011100
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x80000022; // 10000000000000000000000000100010
	const uint32_t loopmask = 0x43fc0001; // 01000011111111000000000000000001
	const uint32_t mval2 = mval | (~Qtp1val & fixmask) | (rng() & loopmask);
	const uint32_t Qtp1valmask2 = Qtp1valmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [140100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x03800000; // 00000011100000000000000000000000
		// [140150] start inner main loop
		uint32_t freebitsval_inner = 0;
		do
		{
			--freebitsval_inner; freebitsval_inner &= 0x407c0001; // 01000000011111000000000000000001
			// [140200] compute value
			uint32_t m = (mval2^freebitsval_inner^freebitsval_outer);
			uint32_t Qtp1 = precomp + m;
			if ( (Qtp1^Qtp1val) & Qtp1valmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (Qtp1^Qtp1val) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [140400] update msgbitrels
			msgbitrel_precomp[12 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc1040001)&1)<<14;
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82940001)&1)<<17;
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41280001)&1)<<18;
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03180002)&1)<<19;
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41bc0000)&1)<<20;
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41140000)&1)<<21;
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02800000)&1)<<22;
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000000)&1)<<23;
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02540001)&1)<<24;
			msgbitrel_precomp[12 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000001)&1)<<31;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x42e40003)&1)<<11;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82d40003)&1)<<12;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00b40001)&1)<<13;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc1c40000)&1)<<14;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x43880001)&1)<<15;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc3440002)&1)<<16;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03680002)&1)<<17;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40140001)&1)<<18;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00800001)&1)<<19;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x401c0000)&1)<<20;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x42540002)&1)<<21;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000001)&1)<<22;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000001)&1)<<30;
			msgbitrel_precomp[12 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000001)&1)<<31;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41480000)&1)<<2;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc2440003)&1)<<11;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc0300001)&1)<<13;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40340000)&1)<<14;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01d80000)&1)<<15;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41dc0001)&1)<<16;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x020c0001)&1)<<17;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc2200001)&1)<<18;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40400000)&1)<<19;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01c80001)&1)<<20;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00480001)&1)<<21;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000001)&1)<<22;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000001)&1)<<23;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41480002)&1)<<24;
			msgbitrel_precomp[12 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00540002)&1)<<30;
			
			// [140699] 
			if (++successcount11 == (1<<2) && 0 == successcount15) return;
			// [140700] call next step
			step12();
		// [140800] end inner main loop
		} while (freebitsval_inner != 0);
	// [140850] end outer main loop
	} while (freebitsval_outer != 0);
// [140900] 
}


// [149997] 
uint32_t successcount10 = 0;
// [150000] step10 forward
void step10()
{
	// [150001] define step number as constant
	static const int t = 10;
	// [150002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [150003] 
	if (t != mainblockoffset+15) successcount10 = 0;
	// [150040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0x70000020; // 01110000000000000000000000100000
	const uint32_t mvalmask = 0x3c001fdc; // 00111100000000000001111111011100
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x40000020; // 01000000000000000000000000100000
	const uint32_t loopmask = 0x83ffe003; // 10000011111111111110000000000011
	const uint32_t mval2 = mval | (~Qtp1val & fixmask) | (rng() & loopmask);
	const uint32_t Qtp1valmask2 = Qtp1valmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [150100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x03000000; // 00000011000000000000000000000000
		// [150150] start inner main loop
		uint32_t freebitsval_inner = 0;
		do
		{
			--freebitsval_inner; freebitsval_inner &= 0x80ffe003; // 10000000111111111110000000000011
			// [150200] compute value
			uint32_t m = (mval2^freebitsval_inner^freebitsval_outer);
			uint32_t Qtp1 = precomp + m;
			if ( (Qtp1^Qtp1val) & Qtp1valmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (Qtp1^Qtp1val) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [150400] update msgbitrels
			msgbitrel_precomp[11 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[11 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[11 - mainblockoffset][12 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000000)&1)<<31;
			msgbitrel_precomp[11 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[11 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41770001)&1)<<14;
			msgbitrel_precomp[11 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x43600000)&1)<<17;
			msgbitrel_precomp[11 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc2c00003)&1)<<18;
			msgbitrel_precomp[11 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00300001)&1)<<19;
			msgbitrel_precomp[11 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03ab0000)&1)<<20;
			msgbitrel_precomp[11 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x031b0002)&1)<<21;
			msgbitrel_precomp[11 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x011b0003)&1)<<24;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03e60003)&1)<<11;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc364c002)&1)<<12;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02628003)&1)<<13;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x42f30003)&1)<<14;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc1e60001)&1)<<15;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02d70001)&1)<<16;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02280003)&1)<<17;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x822b0002)&1)<<18;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82600000)&1)<<19;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x806b0003)&1)<<20;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x809b0002)&1)<<21;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x43000000)&1)<<22;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<23;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000001)&1)<<30;
			msgbitrel_precomp[11 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc0000002)&1)<<31;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000001)&1)<<0;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<1;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01b00001)&1)<<2;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40322000)&1)<<11;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x8208a003)&1)<<13;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x010a4000)&1)<<14;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40898003)&1)<<15;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc15e0001)&1)<<16;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40210002)&1)<<17;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc1140003)&1)<<18;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc2280000)&1)<<19;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81e00002)&1)<<20;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41100001)&1)<<21;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81400000)&1)<<22;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc2800003)&1)<<23;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40b00002)&1)<<24;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<25;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x011b0002)&1)<<30;
			msgbitrel_precomp[11 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000002)&1)<<31;
			
			// [150699] 
			if (++successcount10 == (1<<2) && 0 == successcount15) return;
			// [150700] call next step
			step11();
		// [150800] end inner main loop
		} while (freebitsval_inner != 0);
	// [150850] end outer main loop
	} while (freebitsval_outer != 0);
// [150900] 
}


// [159997] 
uint32_t successcount9 = 0;
// [160000] step9 forward
void step9()
{
	// [160001] define step number as constant
	static const int t = 9;
	// [160002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [160003] 
	if (t != mainblockoffset+15) successcount9 = 0;
	// [160040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0xd4000088; // 11010100000000000000000010001000
	const uint32_t mvalmask = 0x5000009c; // 01010000000000000000000010011100
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000014; // 00000000000000000000000000010100
	const uint32_t loopmask = 0x2bffff63; // 00101011111111111111111101100011
	const uint32_t Qtp1val2 = Qtp1val | (~mval & fixmask) | (rng() & loopmask);
	const uint32_t mvalmask2 = mvalmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [160100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x2b000063; // 00101011000000000000000001100011
		// [160150] start inner main loop
		uint32_t freebitsval_inner = 0;
		do
		{
			--freebitsval_inner; freebitsval_inner &= 0x00ffff00; // 00000000111111111111111100000000
			// [160200] compute value
			uint32_t Qtp1 = (Qtp1val2^freebitsval_inner^freebitsval_outer);
			uint32_t m = Qtp1 - precomp;
			if ( (m^mval) & mvalmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (m^mval) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [160400] update msgbitrels
			msgbitrel_precomp[10 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[10 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[10 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[10 - mainblockoffset][12 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000001)&1)<<31;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x093fc000)&1)<<14;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x89980000)&1)<<17;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x23300003)&1)<<18;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x86ac0000)&1)<<19;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x8100c003)&1)<<20;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x8c8cc002)&1)<<21;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x0b000000)&1)<<22;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x2e000000)&1)<<23;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x890cc003)&1)<<24;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x08000000)&1)<<25;
			msgbitrel_precomp[10 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<31;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000001)&1)<<0;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<1;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x29387003)&1)<<11;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x0b4c5002)&1)<<12;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x83586001)&1)<<13;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x84a94000)&1)<<14;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x29528003)&1)<<15;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xafa9c002)&1)<<16;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x85860001)&1)<<17;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x8798c001)&1)<<18;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x0d280003)&1)<<19;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xa790c002)&1)<<20;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x29acc002)&1)<<21;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x21400002)&1)<<22;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02800000)&1)<<23;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x0d000000)&1)<<24;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<25;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000000)&1)<<26;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x08000000)&1)<<27;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x20000000)&1)<<29;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x20000003)&1)<<30;
			msgbitrel_precomp[10 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x20000002)&1)<<31;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x20000000)&1)<<0;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x20cc0003)&1)<<2;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x2ca2c000)&1)<<11;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xa9bb0000)&1)<<13;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xae7ac002)&1)<<14;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x832cc002)&1)<<15;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x88d4c002)&1)<<16;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x8670c000)&1)<<17;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x8f600003)&1)<<18;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x26c00001)&1)<<19;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x254c0002)&1)<<20;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x2bcc0002)&1)<<21;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x2e000003)&1)<<22;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000003)&1)<<23;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80cc0000)&1)<<24;
			msgbitrel_precomp[10 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x850cc002)&1)<<30;
			
			// [160699] 
			if (++successcount9 == (1<<2) && 0 == successcount15) return;
			// [160700] call next step
			step10();
		// [160800] end inner main loop
		} while (freebitsval_inner != 0);
	// [160850] end outer main loop
	} while (freebitsval_outer != 0);
// [160900] 
}


// [169997] 
uint32_t successcount8 = 0;
// [170000] step8 forward
void step8()
{
	// [170001] define step number as constant
	static const int t = 8;
	// [170002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [170003] 
	if (t != mainblockoffset+15) successcount8 = 0;
	// [170040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0x50000259; // 01010000000000000000001001011001
	const uint32_t mvalmask = 0x1c000002; // 00011100000000000000000000000010
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x0c000002; // 00001100000000000000000000000010
	const uint32_t loopmask = 0xa3fffda4; // 10100011111111111111110110100100
	const uint32_t Qtp1val2 = Qtp1val | (~mval & fixmask) | (rng() & loopmask);
	const uint32_t mvalmask2 = mvalmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [170100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x03000000; // 00000011000000000000000000000000
		// [170150] start inner main loop
		uint32_t freebitsval_inner = 0;
		do
		{
			--freebitsval_inner; freebitsval_inner &= 0xa0fffda4; // 10100000111111111111110110100100
			// [170200] compute value
			uint32_t Qtp1 = (Qtp1val2^freebitsval_inner^freebitsval_outer);
			uint32_t m = Qtp1 - precomp;
			if ( (m^mval) & mvalmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (m^mval) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [170400] update msgbitrels
			msgbitrel_precomp[9 - mainblockoffset][9 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][9 - mainblockoffset];
			msgbitrel_precomp[9 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[9 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[9 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[9 - mainblockoffset][12 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc0000004)&1)<<31;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<2;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<3;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00bf100c)&1)<<14;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xe0ca0009)&1)<<17;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x61940004)&1)<<18;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x014d0009)&1)<<19;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x60135000)&1)<<20;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc0865004)&1)<<21;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01400000)&1)<<22;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02800000)&1)<<23;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xe1265005)&1)<<24;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<25;
			msgbitrel_precomp[9 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000000)&1)<<31;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x226b9a08)&1)<<11;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x43a08005)&1)<<12;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xe1025000)&1)<<13;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x4248000c)&1)<<14;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40900000)&1)<<15;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81065001)&1)<<16;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xa025000d)&1)<<17;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00a65000)&1)<<18;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc1000004)&1)<<19;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40435000)&1)<<20;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40265009)&1)<<21;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000004)&1)<<22;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000005)&1)<<30;
			msgbitrel_precomp[9 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x20000005)&1)<<31;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<0;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82650008)&1)<<2;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xa0dc3004)&1)<<11;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82e4c00c)&1)<<13;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xe1efd000)&1)<<14;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41d05000)&1)<<15;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x43005001)&1)<<16;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x800f5000)&1)<<17;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80980008)&1)<<18;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81300000)&1)<<19;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x60050000)&1)<<20;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x22a50004)&1)<<21;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc1800004)&1)<<22;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x63000004)&1)<<23;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xa0650009)&1)<<24;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x2026500c)&1)<<30;
			msgbitrel_precomp[9 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000001)&1)<<31;
			
			// [170699] 
			if (++successcount8 == (1<<2) && 0 == successcount15) return;
			// [170700] call next step
			step9();
		// [170800] end inner main loop
		} while (freebitsval_inner != 0);
	// [170850] end outer main loop
	} while (freebitsval_outer != 0);
// [170900] 
}


// [179997] 
uint32_t successcount7 = 0;
// [180000] step7 forward
void step7()
{
	// [180001] define step number as constant
	static const int t = 7;
	// [180002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [180003] 
	if (t != mainblockoffset+15) successcount7 = 0;
	// [180040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0x14000075; // 00010100000000000000000001110101
	const uint32_t mvalmask = 0x7c00281d; // 01111100000000000010100000011101
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000060; // 00000000000000000000000001100000
	const uint32_t loopmask = 0x83ffd782; // 10000011111111111101011110000010
	const uint32_t mval2 = mval | (~Qtp1val & fixmask) | (rng() & loopmask);
	const uint32_t Qtp1valmask2 = Qtp1valmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [180100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x03c00002; // 00000011110000000000000000000010
		// [180150] start inner main loop
		uint32_t freebitsval_inner = 0;
		do
		{
			--freebitsval_inner; freebitsval_inner &= 0x803fd780; // 10000000001111111101011110000000
			// [180200] compute value
			uint32_t m = (mval2^freebitsval_inner^freebitsval_outer);
			uint32_t Qtp1 = precomp + m;
			if ( (Qtp1^Qtp1val) & Qtp1valmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (Qtp1^Qtp1val) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [180400] update msgbitrels
			msgbitrel_precomp[8 - mainblockoffset][8 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][8 - mainblockoffset];
			msgbitrel_precomp[8 - mainblockoffset][9 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][9 - mainblockoffset];
			msgbitrel_precomp[8 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[8 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[8 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[8 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[8 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x017b0002)&1)<<14;
			msgbitrel_precomp[8 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00e00000)&1)<<17;
			msgbitrel_precomp[8 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01c00000)&1)<<18;
			msgbitrel_precomp[8 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81f00002)&1)<<19;
			msgbitrel_precomp[8 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03970002)&1)<<20;
			msgbitrel_precomp[8 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80e70002)&1)<<21;
			msgbitrel_precomp[8 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82e70002)&1)<<24;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00d7c000)&1)<<11;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03e04002)&1)<<12;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03578002)&1)<<13;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x03610000)&1)<<14;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02c20000)&1)<<15;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x83630000)&1)<<16;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01780002)&1)<<17;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80f70002)&1)<<18;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00200000)&1)<<19;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80d70002)&1)<<20;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02670000)&1)<<21;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81000000)&1)<<22;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<23;
			msgbitrel_precomp[8 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000000)&1)<<30;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<0;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82700000)&1)<<2;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81921002)&1)<<11;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x83884000)&1)<<13;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81f78002)&1)<<14;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82b60002)&1)<<15;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02a50002)&1)<<16;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00130002)&1)<<17;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01080000)&1)<<18;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82100000)&1)<<19;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82500000)&1)<<20;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82300000)&1)<<21;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00800000)&1)<<22;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000000)&1)<<23;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80700002)&1)<<24;
			msgbitrel_precomp[8 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82e70002)&1)<<30;
			
			// [180699] 
			if (++successcount7 == (1<<2) && 0 == successcount15) return;
			// [180700] call next step
			step8();
		// [180800] end inner main loop
		} while (freebitsval_inner != 0);
	// [180850] end outer main loop
	} while (freebitsval_outer != 0);
// [180900] 
}


// [189997] 
uint32_t successcount6 = 0;
// [190000] step6 forward
void step6()
{
	// [190001] define step number as constant
	static const int t = 6;
	// [190002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [190003] 
	if (t != mainblockoffset+15) successcount6 = 0;
	// [190040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0x7c0001fd; // 01111100000000000000000111111101
	const uint32_t mvalmask = 0x3c0001dc; // 00111100000000000000000111011100
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t loopmask = 0x83fffe02; // 10000011111111111111111000000010
	const uint32_t Qtp1val2 = Qtp1val | (~mval & fixmask) | (rng() & loopmask);
	const uint32_t mvalmask2 = mvalmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [190100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x03c00002; // 00000011110000000000000000000010
		// [190150] start inner main loop
		uint32_t freebitsval_inner = 0;
		do
		{
			--freebitsval_inner; freebitsval_inner &= 0x803ffe00; // 10000000001111111111111000000000
			// [190200] compute value
			uint32_t Qtp1 = (Qtp1val2^freebitsval_inner^freebitsval_outer);
			uint32_t m = Qtp1 - precomp;
			if ( (m^mval) & mvalmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (m^mval) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [190400] update msgbitrels
			msgbitrel_precomp[7 - mainblockoffset][7 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][7 - mainblockoffset];
			msgbitrel_precomp[7 - mainblockoffset][8 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][8 - mainblockoffset];
			msgbitrel_precomp[7 - mainblockoffset][9 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][9 - mainblockoffset];
			msgbitrel_precomp[7 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[7 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[7 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[7 - mainblockoffset][12 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000001)&1)<<31;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x83094001)&1)<<14;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41080001)&1)<<17;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc2100000)&1)<<18;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x42a40002)&1)<<19;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x836c4003)&1)<<20;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc1284002)&1)<<21;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000000)&1)<<22;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<23;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41a84003)&1)<<24;
			msgbitrel_precomp[7 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<31;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<0;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc0645802)&1)<<11;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc1b02000)&1)<<12;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x804c0001)&1)<<13;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x83c88001)&1)<<14;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc3910002)&1)<<15;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x828a4002)&1)<<16;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00c00003)&1)<<17;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc1204001)&1)<<18;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01100003)&1)<<19;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc10c4001)&1)<<20;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81e84003)&1)<<21;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80800002)&1)<<22;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01000000)&1)<<23;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<24;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000003)&1)<<30;
			msgbitrel_precomp[7 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x40000003)&1)<<31;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc2840000)&1)<<2;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x8135c002)&1)<<11;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc2c70003)&1)<<13;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc0264000)&1)<<14;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00304003)&1)<<15;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc3904000)&1)<<16;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x435c4000)&1)<<17;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80e00002)&1)<<18;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x81c00001)&1)<<19;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41040002)&1)<<20;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x01840003)&1)<<21;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000003)&1)<<22;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0xc0000000)&1)<<23;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82840001)&1)<<24;
			msgbitrel_precomp[7 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x41a84000)&1)<<30;
			
			// [190699] 
			if (++successcount6 == (1<<2) && 0 == successcount15) return;
			// [190700] call next step
			step7();
		// [190800] end inner main loop
		} while (freebitsval_inner != 0);
	// [190850] end outer main loop
	} while (freebitsval_outer != 0);
// [190900] 
}


// [199997] 
uint32_t successcount5 = 0;
// [200000] step5 forward
void step5()
{
	// [200001] define step number as constant
	static const int t = 5;
	// [200002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [200003] 
	if (t != mainblockoffset+15) successcount5 = 0;
	// [200040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0xf5fffdf7; // 11110101111111111111110111110111
	const uint32_t mvalmask = 0x69f81015; // 01101001111110000001000000010101
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x08000000; // 00001000000000000000000000000000
	const uint32_t loopmask = 0x02000208; // 00000010000000000000001000001000
	const uint32_t Qtp1val2 = Qtp1val | (~mval & fixmask) | (rng() & loopmask);
	const uint32_t mvalmask2 = mvalmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [200100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x02000208; // 00000010000000000000001000001000
		// [200150] start inner main loop
		for (uint32_t freebitsval_inner = 0; freebitsval_inner == 0; ++freebitsval_inner)
		{
			// [200200] compute value
			uint32_t Qtp1 = (Qtp1val2^freebitsval_inner^freebitsval_outer);
			uint32_t m = Qtp1 - precomp;
			if ( (m^mval) & mvalmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (m^mval) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [200400] update msgbitrels
			msgbitrel_precomp[6 - mainblockoffset][6 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][6 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][7 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][7 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][7 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x10000000)&1)<<28;
			msgbitrel_precomp[6 - mainblockoffset][8 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][8 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][9 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][9 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][11 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<3;
			msgbitrel_precomp[6 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x1206a008)&1)<<14;
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x9604000a)&1)<<17;
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x14000000)&1)<<18;
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x0002000a)&1)<<19;
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x92042000)&1)<<20;
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x12062000)&1)<<21;
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04062002)&1)<<24;
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000000)&1)<<25;
			msgbitrel_precomp[6 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x10000000)&1)<<27;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04006408)&1)<<11;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x12000002)&1)<<12;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82042000)&1)<<13;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80040008)&1)<<14;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<15;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x10062002)&1)<<16;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x9602000a)&1)<<17;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x16062000)&1)<<18;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x14042000)&1)<<20;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x1406200a)&1)<<21;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x94000000)&1)<<22;
			msgbitrel_precomp[6 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000000)&1)<<30;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<0;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82020008)&1)<<2;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x90060000)&1)<<11;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x0200400a)&1)<<13;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x1206a000)&1)<<14;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x14052000)&1)<<15;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x94042002)&1)<<16;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x92002000)&1)<<17;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x86000008)&1)<<18;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x94000000)&1)<<19;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82020000)&1)<<20;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x82020000)&1)<<21;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x10000000)&1)<<22;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x0002000a)&1)<<24;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000000)&1)<<25;
			msgbitrel_precomp[6 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x06062008)&1)<<30;
			
			// [200699] 
			if (++successcount5 == (1<<2) && 0 == successcount15) return;
			// [200700] call next step
			step6();
		// [200800] end inner main loop
		};
	// [200850] end outer main loop
	} while (freebitsval_outer != 0);
// [200900] 
}


// [209997] 
uint32_t successcount4 = 0;
// [210000] step4 forward
void step4()
{
	// [210001] define step number as constant
	static const int t = 4;
	// [210002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [210003] 
	if (t != mainblockoffset+15) successcount4 = 0;
	// [210040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0x7ffffcc5; // 01111111111111111111110011000101
	const uint32_t mvalmask = 0x7ffa005b; // 01111111111110100000000001011011
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x0000001a; // 00000000000000000000000000011010
	const uint32_t loopmask = 0x80000320; // 10000000000000000000001100100000
	const uint32_t Qtp1val2 = Qtp1val | (~mval & fixmask) | (rng() & loopmask);
	const uint32_t mvalmask2 = mvalmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [210100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x00000020; // 00000000000000000000000000100000
		// [210150] start inner main loop
		uint32_t freebitsval_inner = 0;
		do
		{
			--freebitsval_inner; freebitsval_inner &= 0x80000300; // 10000000000000000000001100000000
			// [210200] compute value
			uint32_t Qtp1 = (Qtp1val2^freebitsval_inner^freebitsval_outer);
			uint32_t m = Qtp1 - precomp;
			if ( (m^mval) & mvalmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (m^mval) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [210400] update msgbitrels
			msgbitrel_precomp[5 - mainblockoffset][5 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][5 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][6 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][6 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][7 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][7 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][8 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][8 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][9 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][9 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][12 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<31;
			msgbitrel_precomp[5 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00050004)&1)<<14;
			msgbitrel_precomp[5 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<18;
			msgbitrel_precomp[5 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80010004)&1)<<20;
			msgbitrel_precomp[5 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80010004)&1)<<21;
			msgbitrel_precomp[5 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80010004)&1)<<24;
			msgbitrel_precomp[5 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<31;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<1;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x8001d004)&1)<<11;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80056000)&1)<<12;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x0001c004)&1)<<13;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00058004)&1)<<14;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00010000)&1)<<15;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00050000)&1)<<16;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00040000)&1)<<17;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80010004)&1)<<18;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80010004)&1)<<20;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80010004)&1)<<21;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<23;
			msgbitrel_precomp[5 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<31;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000000)&1)<<2;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80048804)&1)<<11;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80002000)&1)<<13;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80054000)&1)<<14;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80018004)&1)<<15;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000000)&1)<<16;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80010004)&1)<<17;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00040004)&1)<<18;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<19;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<20;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000004)&1)<<21;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<23;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000004)&1)<<24;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80010000)&1)<<30;
			msgbitrel_precomp[5 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x80000000)&1)<<31;
			
			// [210699] 
			if (++successcount4 == (1<<2) && 0 == successcount15) return;
			// [210700] call next step
			step5();
		// [210800] end inner main loop
		} while (freebitsval_inner != 0);
	// [210850] end outer main loop
	} while (freebitsval_outer != 0);
// [210900] 
}


// [219997] 
uint32_t successcount3 = 0;
// [220000] step3 forward
void step3()
{
	// [220001] define step number as constant
	static const int t = 3;
	// [220002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [220003] 
	if (t != mainblockoffset+15) successcount3 = 0;
	// [220040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0x7ffff7f7; // 01111111111111111111011111110111
	const uint32_t mvalmask = 0x7fffc1e7; // 01111111111111111100000111100111
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t loopmask = 0x80000808; // 10000000000000000000100000001000
	const uint32_t Qtp1val2 = Qtp1val | (~mval & fixmask) | (rng() & loopmask);
	const uint32_t mvalmask2 = mvalmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [220100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x00000808; // 00000000000000000000100000001000
		// [220150] start inner main loop
		uint32_t freebitsval_inner = 0;
		do
		{
			--freebitsval_inner; freebitsval_inner &= 0x80000000; // 10000000000000000000000000000000
			// [220200] compute value
			uint32_t Qtp1 = (Qtp1val2^freebitsval_inner^freebitsval_outer);
			uint32_t m = Qtp1 - precomp;
			if ( (m^mval) & mvalmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (m^mval) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [220400] update msgbitrels
			msgbitrel_precomp[4 - mainblockoffset][4 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][4 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][5 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][5 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][6 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][6 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][7 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][7 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][7 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<3;
			msgbitrel_precomp[4 - mainblockoffset][8 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][8 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][9 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][9 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][9 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<3;
			msgbitrel_precomp[4 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][11 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<3;
			msgbitrel_precomp[4 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002008)&1)<<14;
			msgbitrel_precomp[4 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<17;
			msgbitrel_precomp[4 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<19;
			msgbitrel_precomp[4 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<20;
			msgbitrel_precomp[4 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<21;
			msgbitrel_precomp[4 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<24;
			msgbitrel_precomp[4 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000408)&1)<<11;
			msgbitrel_precomp[4 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00001000)&1)<<12;
			msgbitrel_precomp[4 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<14;
			msgbitrel_precomp[4 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<16;
			msgbitrel_precomp[4 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<17;
			msgbitrel_precomp[4 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<18;
			msgbitrel_precomp[4 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<20;
			msgbitrel_precomp[4 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002008)&1)<<21;
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<2;
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<13;
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<14;
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<15;
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<16;
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002000)&1)<<17;
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<18;
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<24;
			msgbitrel_precomp[4 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00002008)&1)<<30;
			
			// [220699] 
			if (++successcount3 == (1<<2) && 0 == successcount15) return;
			// [220700] call next step
			step4();
		// [220800] end inner main loop
		} while (freebitsval_inner != 0);
	// [220850] end outer main loop
	} while (freebitsval_outer != 0);
// [220900] 
}


// [229997] 
uint32_t successcount2 = 0;
// [230000] step2 forward
void step2()
{
	// [230001] define step number as constant
	static const int t = 2;
	// [230002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [230003] 
	if (t != mainblockoffset+15) successcount2 = 0;
	// [230040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0xffffff7f; // 11111111111111111111111101111111
	const uint32_t mvalmask = 0xfffff83f; // 11111111111111111111100000111111
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t loopmask = 0x00000080; // 00000000000000000000000010000000
	const uint32_t Qtp1val2 = Qtp1val | (~mval & fixmask) | (rng() & loopmask);
	const uint32_t mvalmask2 = mvalmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [230100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x00000080; // 00000000000000000000000010000000
		// [230150] start inner main loop
		for (uint32_t freebitsval_inner = 0; freebitsval_inner == 0; ++freebitsval_inner)
		{
			// [230200] compute value
			uint32_t Qtp1 = (Qtp1val2^freebitsval_inner^freebitsval_outer);
			uint32_t m = Qtp1 - precomp;
			if ( (m^mval) & mvalmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (m^mval) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [230400] update msgbitrels
			msgbitrel_precomp[3 - mainblockoffset][3 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][3 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][4 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][4 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][5 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][5 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][6 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][6 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][7 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][7 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][8 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][8 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][9 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][9 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[3 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000200)&1)<<11;
			msgbitrel_precomp[3 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			
			// [230699] 
			if (++successcount2 == (1<<2) && 0 == successcount15) return;
			// [230700] call next step
			step3();
		// [230800] end inner main loop
		};
	// [230850] end outer main loop
	} while (freebitsval_outer != 0);
// [230900] 
}


// [239997] 
uint32_t successcount1 = 0;
// [240000] step1 forward
void step1()
{
	// [240001] define step number as constant
	static const int t = 1;
	// [240002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [240003] 
	if (t != mainblockoffset+15) successcount1 = 0;
	// [240040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0xffffffdd; // 11111111111111111111111111011101
	const uint32_t mvalmask = 0xf9fcf09d; // 11111001111111001111000010011101
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t loopmask = 0x00000022; // 00000000000000000000000000100010
	const uint32_t Qtp1val2 = Qtp1val | (~mval & fixmask) | (rng() & loopmask);
	const uint32_t mvalmask2 = mvalmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [240100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x00000022; // 00000000000000000000000000100010
		// [240150] start inner main loop
		for (uint32_t freebitsval_inner = 0; freebitsval_inner == 0; ++freebitsval_inner)
		{
			// [240200] compute value
			uint32_t Qtp1 = (Qtp1val2^freebitsval_inner^freebitsval_outer);
			uint32_t m = Qtp1 - precomp;
			if ( (m^mval) & mvalmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (m^mval) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [240400] update msgbitrels
			msgbitrel_precomp[2 - mainblockoffset][2 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][2 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][3 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][3 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][4 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][4 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][5 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][5 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][6 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][6 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][7 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][7 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][8 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][8 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][9 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][9 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x06000000)&1)<<14;
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000002)&1)<<18;
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000000)&1)<<19;
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00010002)&1)<<20;
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02010002)&1)<<21;
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000000)&1)<<22;
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<23;
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04010002)&1)<<24;
			msgbitrel_precomp[2 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<31;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<1;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02010002)&1)<<11;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000002)&1)<<12;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02010000)&1)<<13;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x06000000)&1)<<14;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000002)&1)<<15;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000002)&1)<<16;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04020000)&1)<<17;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00010000)&1)<<18;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<19;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02010002)&1)<<20;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04010002)&1)<<21;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<22;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000000)&1)<<25;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000000)&1)<<26;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<30;
			msgbitrel_precomp[2 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<31;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<2;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00030000)&1)<<13;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00030002)&1)<<14;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00010002)&1)<<15;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02010002)&1)<<16;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x06010000)&1)<<17;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x06000002)&1)<<18;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000000)&1)<<19;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000002)&1)<<20;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x02000002)&1)<<21;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x06000002)&1)<<22;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x04000002)&1)<<23;
			msgbitrel_precomp[2 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00010002)&1)<<30;
			
			// [240699] 
			if (++successcount1 == (1<<2) && 0 == successcount15) return;
			// [240700] call next step
			step2();
		// [240800] end inner main loop
		};
	// [240850] end outer main loop
	} while (freebitsval_outer != 0);
// [240900] 
}


// [249997] 
uint32_t successcount0 = 0;
// [250000] step0 forward firststep
void step0()
{
	// [250001] define step number as constant
	static const int t = 0;
	// [250002] optionally perform statistics
	PERF_ENTER_STEP(t); static bool firsttime = true; if (firsttime) { std::cout << " " << t << std::flush; firsttime = false; }
	// [250003] 
	if (t != mainblockoffset+15) successcount0 = 0;
	// [250040] precompute values
	// Qvalmask: 1-bit means condition on bit of Q, Qval contains target value for those bits, mvalmask & mval similar
	const uint32_t Qtp1valmask = 0xffefe7f7; // 11111111111011111110011111110111
	const uint32_t mvalmask = 0xffcfc7e7; // 11111111110011111100011111100111
	const uint32_t Qtp1val = Qset1mask[Qoffset+t+1] ^ (Qprevmask[Qoffset+t+1] & main_Q1[Qoffset+t]) ^ (Qprevrmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t],30))  ^ (Qprev2rmask[Qoffset+t+1] & rotate_left(main_Q1[Qoffset+t-1],30));
	const uint32_t mval = msgbitrel_precomp[t-mainblockoffset][t-mainblockoffset];
	const uint32_t fixmask = 0x00000000; // 00000000000000000000000000000000
	const uint32_t loopmask = 0x00101808; // 00000000000100000001100000001000
	const uint32_t Qtp1val2 = Qtp1val | (~mval & fixmask) | (rng() & loopmask);
	const uint32_t mvalmask2 = mvalmask & ~fixmask;
	const uint32_t precomp = rotate_left(main_Q1[Qoffset+t],5) + sha1_f1(main_Q1[Qoffset+t-1],main_Q1r30[Qoffset+t-2],main_Q1r30[Qoffset+t-3]) + rotate_left(main_Q1[Qoffset+t-4],30) + sha1_ac[0];
	// [250100] start outer main loop
	uint32_t freebitsval_outer = 0;
	do
	{
		--freebitsval_outer; freebitsval_outer &= 0x00101808; // 00000000000100000001100000001000
		// [250150] start inner main loop
		for (uint32_t freebitsval_inner = 0; freebitsval_inner == 0; ++freebitsval_inner)
		{
			// [250200] compute value
			uint32_t Qtp1 = (Qtp1val2^freebitsval_inner^freebitsval_outer);
			uint32_t m = Qtp1 - precomp;
			if ( (m^mval) & mvalmask2 ) break; // some non-trivial-to-fix condition does not hold => go back to outer loop
			uint32_t fix = (m^mval) & fixmask;
			main_m1   [t]         = m ^ fix;
			main_Q1   [Qoffset+t+1] = Qtp1 ^ fix;
			main_Q1r30[Qoffset+t+1] = rotate_left(main_Q1[Qoffset+t+1],30);
			// [250400] update msgbitrels
			msgbitrel_precomp[1 - mainblockoffset][1 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][1 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][2 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][2 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][3 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][3 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][4 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][4 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][5 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][5 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][6 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][6 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][7 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][7 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][8 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][8 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][9 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][9 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][10 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][10 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][11 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][11 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][12 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][12 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][13 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][13 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<3;
			msgbitrel_precomp[1 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00201008)&1)<<14;
			msgbitrel_precomp[1 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<17;
			msgbitrel_precomp[1 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00200008)&1)<<19;
			msgbitrel_precomp[1 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00001000)&1)<<20;
			msgbitrel_precomp[1 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00101000)&1)<<21;
			msgbitrel_precomp[1 - mainblockoffset][13 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00301000)&1)<<24;
			msgbitrel_precomp[1 - mainblockoffset][14 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][14 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00001008)&1)<<11;
			msgbitrel_precomp[1 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00101000)&1)<<13;
			msgbitrel_precomp[1 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<14;
			msgbitrel_precomp[1 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00201000)&1)<<16;
			msgbitrel_precomp[1 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00000008)&1)<<17;
			msgbitrel_precomp[1 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00301000)&1)<<18;
			msgbitrel_precomp[1 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00101000)&1)<<20;
			msgbitrel_precomp[1 - mainblockoffset][14 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00301008)&1)<<21;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] = msgbitrel_precomp[t - mainblockoffset][15 - mainblockoffset];
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00200008)&1)<<2;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00201000)&1)<<11;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00300008)&1)<<13;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00101000)&1)<<14;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00101000)&1)<<15;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00301000)&1)<<16;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00001000)&1)<<17;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00200008)&1)<<18;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00100000)&1)<<19;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00200000)&1)<<21;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00200008)&1)<<24;
			msgbitrel_precomp[1 - mainblockoffset][15 - mainblockoffset] ^= (hamming_weight(main_m1[t] & 0x00301008)&1)<<30;
			
			// [250699] 
			if (++successcount0 == (1<<2) && 0 == successcount15) return;
			// [250700] call next step
			step1();
		// [250800] end inner main loop
		};
	// [250850] end outer main loop
	} while (freebitsval_outer != 0);
// [250900] 
}


// [100000000] start attack
void gen_q13sols()
{
	// [100000002] optionally perform statistics
	PERF_ENTER_STEP(-1);
	
	// [100000010] initialize msgbitrel_precomp
	msgbitrel_precomp[0][0] = 0x304707e1; // 00110000010001110000011111100001
	msgbitrel_precomp[0][1] = 0xd0109089; // 11010000000100001001000010001001
	msgbitrel_precomp[0][2] = 0xe12ef03c; // 11100001001011101111000000111100
	msgbitrel_precomp[0][3] = 0x142bc125; // 00010100001010111100000100100101
	msgbitrel_precomp[0][4] = 0x42a00009; // 01000010101000000000000000001001
	msgbitrel_precomp[0][5] = 0x08b01005; // 00001000101100000001000000000101
	msgbitrel_precomp[0][6] = 0x28000080; // 00101000000000000000000010000000
	msgbitrel_precomp[0][7] = 0x6c002814; // 01101100000000000010100000010100
	msgbitrel_precomp[0][8] = 0x04000002; // 00000100000000000000000000000010
	msgbitrel_precomp[0][9] = 0x00000014; // 00000000000000000000000000010100
	msgbitrel_precomp[0][10] = 0x0c001010; // 00001100000000000001000000010000
	msgbitrel_precomp[0][11] = 0x1800005c; // 00011000000000000000000001011100
	msgbitrel_precomp[0][12] = 0x20000008; // 00100000000000000000000000001000
	msgbitrel_precomp[0][13] = 0x29980018; // 00101001100110000000000000011000
	msgbitrel_precomp[0][14] = 0x1680b029; // 00010110100000001011000000101001
	msgbitrel_precomp[0][15] = 0x05268004; // 00000101001001101000000000000100
	
	// [100000100] start random start state loop
	while (true)
	{
		// [100000200] initialize start state
		main_Q1[Qoffset+-4] = (rng() & ~Qcondmask[Qoffset+-4]) ^ Qset1mask[Qoffset+-4];
		main_Q1[Qoffset+-3] = (rng() & ~Qcondmask[Qoffset+-3]) ^ Qset1mask[Qoffset+-3] ^ (Qprevmask[Qoffset+-3] & main_Q1[Qoffset+-4]);
		main_Q1[Qoffset+-2] = (rng() & ~Qcondmask[Qoffset+-2]) ^ Qset1mask[Qoffset+-2] ^ (Qprevmask[Qoffset+-2] & main_Q1[Qoffset+-3]);
		main_Q1[Qoffset+-1] = (rng() & ~Qcondmask[Qoffset+-1]) ^ Qset1mask[Qoffset+-1] ^ (Qprevmask[Qoffset+-1] & main_Q1[Qoffset+-2]);
		main_Q1[Qoffset+0] = (rng() & ~Qcondmask[Qoffset+0]) ^ Qset1mask[Qoffset+0] ^ (Qprevmask[Qoffset+0] & main_Q1[Qoffset+-1]);
		main_Q1r30[Qoffset+-4] = rotate_left(main_Q1[Qoffset+-4],30);
		main_Q1r30[Qoffset+-3] = rotate_left(main_Q1[Qoffset+-3],30);
		main_Q1r30[Qoffset+-2] = rotate_left(main_Q1[Qoffset+-2],30);
		main_Q1r30[Qoffset+-1] = rotate_left(main_Q1[Qoffset+-1],30);
		main_Q1r30[Qoffset+0] = rotate_left(main_Q1[Qoffset+0],30);
		
		// [100000300] call first step
		step0();
		
	// [100000800] end random start state loop
	}
	
// [100000900] end function start_attack()
}


// [1000000000] main function
/*
int main(int argc, char** argv)
{
        const char* rndseedchars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.+";
        std::string rndseed;
        if (argc >= 2)
        {
            rndseed = std::string(argv[1]);
        }
        else
        {
            for (int i = 0; i < 32; ++i)
                rndseed += rndseedchars[xrng128() % 64];
        }
        std::cout << "Seed: " << rndseed << std::endl;
        seed(0);
        for (int i = 0; i < rndseed.size(); ++i)
            addseed( uint32_t(rndseed[i]) ^ (uint32_t(rndseed[i])*4864) );
        gen_q13sols();
	return 0;
}
*/
/*
===== Q14 ======
W10[11] w15[11]fwQ12 0.0732422 fwQ13 3.68652 fwQ14 16.5283 Q14[1]1.7334 Q15[27]9.10645 Q15[29]3.58887
for (int W10t11b = 0; W10t11b < 2; ++W10t11b) { m1[10] ^= 0x800; m1[15] ^= 0x800;  
W11[16] w15[13]fwQ12 0.0976562 fwQ13 3.80859 fwQ14 17.041 Q14[1]1.92871 Q15[27]7.10449 Q15[29]2.68555
for (int W11t16b = 0; W11t16b < 2; ++W11t16b) { m1[11] ^= 0x10000; m1[15] ^= 0x2000;  
W11[17] w13[14] w13[19] w13[20] w14[11] w14[13] w14[17] w14[20] w15[2] w15[11] w15[14] w15[15] w15[17] w15[20] w15[21] w15[24]fwQ12 0.195312 fwQ13 6.64062 fwQ14 31.0791 Q14[1]2.83203 Q15[29]8.37402 Q15[31]4.32129
for (int W11t17b = 0; W11t17b < 2; ++W11t17b) { m1[11] ^= 0x20000; m1[13] ^= 0x184000; m1[14] ^= 0x122800; m1[15] ^= 0x132c804;  
W12[0] w13[14] w13[17] w13[24] w14[11] w14[14] w14[16] w15[11] w15[16] w15[18] w15[21]fwQ14 53.5645 Q14[29]6.39648 Q15[27]3.73535
for (int W12t0b = 0; W12t0b < 2; ++W12t0b) { m1[12] ^= 0x1; m1[13] ^= 0x1024000; m1[14] ^= 0x14800; m1[15] ^= 0x250800;  
W12[21] w13[17] w14[15] w14[18] w14[20] w15[13] w15[14] w15[16] w15[17]fwQ13 4.39453 fwQ14 17.2119 Q14[1]2.05078 Q15[27]6.39648 Q15[29]2.44141
for (int W12t21b = 0; W12t21b < 2; ++W12t21b) { m1[12] ^= 0x200000; m1[13] ^= 0x20000; m1[14] ^= 0x148000; m1[15] ^= 0x36000;  
W12[22] w13[18] w13[20] w13[21] w13[24] w14[11] w14[12] w14[13] w14[18] w14[19] w14[20] w15[16] w15[18] w15[30]fwQ13 8.3252 fwQ14 31.9824 Q14[1]3.6377
for (int W12t22b = 0; W12t22b < 2; ++W12t22b) { m1[12] ^= 0x400000; m1[13] ^= 0x1340000; m1[14] ^= 0x1c3800; m1[15] ^= 0x40050000;  
W13[0] w13[17] w13[19] w13[20] w13[21] w14[12] w14[13] w14[14] w14[16] w14[17] w14[18] w14[20] w15[2] w15[13] w15[14] w15[15] w15[17] w15[18] w15[24]fwQ14 50.2441 Q14[29]1.17188 Q15[31]6.29883
for (int W13t0b = 0; W13t0b < 2; ++W13t0b) { m1[13] ^= 0x3a0001; m1[14] ^= 0x177000; m1[15] ^= 0x106e004;  
W13[1] w13[14] w13[17] w13[18] w13[19] w13[20] w13[24] w14[12] w14[13] w14[15] w14[16] w14[20] w14[22] w15[2] w15[11] w15[13] w15[15] w15[17] w15[19] w15[20] w15[21] w15[23] w15[30]fwQ14 100 Q14[29]6.37207
for (int W13t1b = 0; W13t1b < 2; ++W13t1b) { m1[13] ^= 0x11e4002; m1[14] ^= 0x51b000; m1[15] ^= 0x40baa804;  
===== Q15 ======
W10[6]fwQ13 0.268555 fwQ14 1.00098 fwQ15 52.2949 Q15[0]2.14844 Q15[1]1.12305 Q15[31]6.5918 Q16[29]4.49219 Q16[30]4.02832
for (int W10t6b = 0; W10t6b < 2; ++W10t6b) { m1[10] ^= 0x40;  
W10[7]fwQ13 0.341797 fwQ14 1.78223 fwQ15 99.5605 Q15[0]4.02832 Q15[1]2.05078 Q16[29]7.2998 Q16[30]6.5918 Q16[31]6.68945
for (int W10t7b = 0; W10t7b < 2; ++W10t7b) { m1[10] ^= 0x80;  
W10[8]fwQ13 0.683594 fwQ14 3.125 fwQ15 55.542 Q15[0]7.54395 Q15[1]4.07715 Q15[27]1.75781
for (int W10t8b = 0; W10t8b < 2; ++W10t8b) { m1[10] ^= 0x100;  
W10[9]fwQ12 0.0244141 fwQ13 1.46484 fwQ14 6.03027 fwQ15 99.9023 Q15[1]7.61719 Q15[27]2.73438 Q16[1]9.05762
for (int W10t9b = 0; W10t9b < 2; ++W10t9b) { m1[10] ^= 0x200;  
W10[10]fwQ13 2.14844 fwQ14 9.66797 fwQ15 64.5996 Q15[27]5.07812 Q15[29]1.78223
for (int W10t10b = 0; W10t10b < 2; ++W10t10b) { m1[10] ^= 0x400;  
W11[9]fwQ13 0.0488281 fwQ14 0.390625 fwQ15 13.5498 Q15[0]0.78125 Q15[1]0.341797 Q15[29]2.92969 Q15[31]1.48926 Q16[1]6.86035 Q16[27]3.83301 Q17[1]8.3252 Q17[31]7.9834
for (int W11t9b = 0; W11t9b < 2; ++W11t9b) { m1[11] ^= 0x200;  
W11[10] w14[11]fwQ13 0.12207 fwQ14 0.610352 fwQ15 24.2432 Q15[0]1.26953 Q15[1]0.585938 Q15[29]5.46875 Q15[31]2.58789 Q16[27]5.27344 Q16[29]2.05078
for (int W11t10b = 0; W11t10b < 2; ++W11t10b) { m1[11] ^= 0x400; m1[14] ^= 0x800;  
W11[11]fwQ13 0.146484 fwQ14 1.02539 fwQ15 50.4639 Q15[0]2.24609 Q15[1]1.07422 Q15[31]5.37109 Q16[29]3.78418 Q16[30]3.58887
for (int W11t11b = 0; W11t11b < 2; ++W11t11b) { m1[11] ^= 0x800;  
W11[12] w14[11]fwQ13 0.268555 fwQ14 1.70898 fwQ15 99.8047 Q15[0]4.12598 Q15[1]1.92871 Q16[29]5.9082 Q16[30]5.32227 Q16[31]5.37109
for (int W11t12b = 0; W11t12b < 2; ++W11t12b) { m1[11] ^= 0x1000; m1[14] ^= 0x800;  
W11[13] w13[14] w13[20] w13[21] w13[24] w14[11] w14[13] w14[16] w14[18] w14[20] w14[21] w15[14] w15[15] w15[16] w15[17] w15[30]fwQ13 0.634766 fwQ14 6.17676 fwQ15 83.5205 Q15[1]6.95801
for (int W11t13b = 0; W11t13b < 2; ++W11t13b) { m1[11] ^= 0x2000; m1[13] ^= 0x1304000; m1[14] ^= 0x352800; m1[15] ^= 0x4003c000;  
W11[14] w15[11]fwQ13 1.00098 fwQ14 5.46875 fwQ15 99.9756 Q15[1]7.15332 Q15[27]2.0752 Q16[1]7.9834 Q17[28]9.86328
for (int W11t14b = 0; W11t14b < 2; ++W11t14b) { m1[11] ^= 0x4000; m1[15] ^= 0x800;  
W11[15] w13[20] w13[21] w13[24] w14[11] w14[13] w14[16] w14[18] w14[20] w14[21] w15[14] w15[15] w15[16] w15[17] w15[30]fwQ13 1.97754 fwQ14 9.61914 fwQ15 99.9268
for (int W11t15b = 0; W11t15b < 2; ++W11t15b) { m1[11] ^= 0x8000; m1[13] ^= 0x1300000; m1[14] ^= 0x352800; m1[15] ^= 0x4003c000;  
W12[2] w13[20] w13[31] w14[1] w14[17] w14[18] w14[19] w14[20] w14[21] w14[22] w14[23] w14[30] w15[2] w15[15] w15[17] w15[18] w15[19] w15[20] w15[21] w15[22] w15[24]fwQ14 0.756836 fwQ15 100 Q15[0]1.34277 Q15[29]6.0791
for (int W12t2b = 0; W12t2b < 2; ++W12t2b) { m1[12] ^= 0x4; m1[13] ^= 0x80100000; m1[14] ^= 0x40fe0002; m1[15] ^= 0x17e8004;  
W12[14] w14[11] w14[13]fwQ13 0.0244141 fwQ14 0.244141 fwQ15 13.2568 Q15[0]0.708008 Q15[1]0.268555 Q15[29]2.9541 Q15[31]1.58691 Q16[1]6.44531 Q16[27]2.56348 Q17[1]7.4707 Q17[31]6.29883
for (int W12t14b = 0; W12t14b < 2; ++W12t14b) { m1[12] ^= 0x4000; m1[14] ^= 0x2800;  
W12[15] w14[12] w14[14] w15[11]fwQ13 0.170898 fwQ14 0.756836 fwQ15 25.8301 Q15[0]1.48926 Q15[1]0.488281 Q15[29]6.32324 Q15[31]2.97852 Q16[27]4.24805 Q16[29]1.75781
for (int W12t15b = 0; W12t15b < 2; ++W12t15b) { m1[12] ^= 0x8000; m1[14] ^= 0x5000; m1[15] ^= 0x800;  
W12[16] w13[14] w13[20] w13[21] w13[24] w14[11] w14[15] w14[16] w14[18] w14[20] w14[21] w15[11] w15[14] w15[15] w15[16] w15[17] w15[30]fwQ13 0.219727 fwQ14 6.66504 fwQ15 98.2178 Q15[1]7.73926
for (int W12t16b = 0; W12t16b < 2; ++W12t16b) { m1[12] ^= 0x10000; m1[13] ^= 0x1304000; m1[14] ^= 0x358800; m1[15] ^= 0x4003c800;  
W12[17] w14[14] w14[16] w15[13]fwQ13 0.390625 fwQ14 1.78223 fwQ15 99.8047 Q15[0]4.05273 Q15[1]1.95312 Q16[29]4.66309 Q16[30]4.41895 Q16[31]4.54102
for (int W12t17b = 0; W12t17b < 2; ++W12t17b) { m1[12] ^= 0x20000; m1[14] ^= 0x14000; m1[15] ^= 0x2000;  
W12[18] w13[14] w14[12] w14[15] w14[17] w15[11] w15[13] w15[14]fwQ13 0.830078 fwQ14 3.51562 fwQ15 55.9082 Q15[0]7.8125 Q15[1]3.83301 Q15[27]1.07422 Q16[29]9.52148 Q16[30]8.81348 Q16[31]9.05762
for (int W12t18b = 0; W12t18b < 2; ++W12t18b) { m1[12] ^= 0x40000; m1[13] ^= 0x4000; m1[14] ^= 0x29000; m1[15] ^= 0x6800;  
W12[19] w14[11] w14[12] w14[13] w14[16] w14[18] w15[11] w15[14] w15[15]fwQ13 1.19629 fwQ14 5.78613 fwQ15 100 Q15[1]7.27539 Q15[27]1.3916 Q16[1]7.83691
for (int W12t19b = 0; W12t19b < 2; ++W12t19b) { m1[12] ^= 0x80000; m1[14] ^= 0x53800; m1[15] ^= 0xc800;  
W12[20] w13[14] w13[19] w13[21] w13[24] w14[11] w14[13] w14[14] w14[16] w14[18] w14[19] w14[21] w15[2] w15[11] w15[13] w15[14] w15[15] w15[20] w15[21] w15[24] w15[30]fwQ13 2.44141 fwQ14 9.74121 fwQ15 100
for (int W12t20b = 0; W12t20b < 2; ++W12t20b) { m1[12] ^= 0x100000; m1[13] ^= 0x1284000; m1[14] ^= 0x2d6800; m1[15] ^= 0x4130e804;  
W13[12] w13[14] w13[20] w13[21] w13[24] w14[11] w14[13] w14[16] w14[18] w14[20] w14[21] w15[14] w15[15] w15[16] w15[17] w15[30]fwQ14 6.64062 fwQ15 100 Q15[1]7.76367
for (int W13t12b = 0; W13t12b < 2; ++W13t12b) { m1[13] ^= 0x1305000; m1[14] ^= 0x352800; m1[15] ^= 0x4003c000;  
W13[16] w13[19] w13[20] w14[17] w14[20] w15[2] w15[11] w15[15] w15[17] w15[20] w15[21] w15[24]fwQ14 0.732422 fwQ15 26.3184 Q15[0]1.34277 Q15[1]0.634766 Q15[29]5.9082 Q15[31]2.90527
for (int W13t16b = 0; W13t16b < 2; ++W13t16b) { m1[13] ^= 0x190000; m1[14] ^= 0x120000; m1[15] ^= 0x1328804;  
===== Q16 ======
W11[7]fwQ14 0.12207 fwQ15 3.78418 fwQ16 99.3896 Q16[1]1.83105 Q17[31]2.51465
for (int W11t7b = 0; W11t7b < 2; ++W11t7b) { m1[11] ^= 0x80;  
W11[8]fwQ13 0.0244141 fwQ14 0.170898 fwQ15 6.93359 fwQ16 52.7832 Q16[1]3.58887 Q16[27]1.7334 Q17[31]4.61426
for (int W11t8b = 0; W11t8b < 2; ++W11t8b) { m1[11] ^= 0x100;  
W12[9]fwQ15 0.512695 fwQ16 13.6475 Q16[1]0.268555 Q16[29]3.39355 Q16[30]3.39355 Q16[31]3.39355 Q17[1]6.5918 Q17[28]1.9043 Q18[31]6.27441
for (int W12t9b = 0; W12t9b < 2; ++W12t9b) { m1[12] ^= 0x200;  
W12[10]fwQ15 0.854492 fwQ16 26.0498 Q16[1]0.512695 Q16[29]6.71387 Q16[30]6.71387 Q16[31]6.71387 Q17[28]3.27148
for (int W12t10b = 0; W12t10b < 2; ++W12t10b) { m1[12] ^= 0x400;  
W12[11]fwQ14 0.0244141 fwQ15 2.05078 fwQ16 51.7334 Q16[1]1.02539 Q17[28]5.98145
for (int W12t11b = 0; W12t11b < 2; ++W12t11b) { m1[12] ^= 0x800;  
W12[12] w14[11]fwQ14 0.146484 fwQ15 3.73535 fwQ16 99.5117 Q16[1]2.05078 Q17[28]9.2041 Q17[31]2.05078
for (int W12t12b = 0; W12t12b < 2; ++W12t12b) { m1[12] ^= 0x1000; m1[14] ^= 0x800;  
W12[13] w14[11] w14[12]fwQ13 0.0244141 fwQ14 0.195312 fwQ15 7.2998 fwQ16 53.9795 Q16[1]3.54004 Q16[27]1.44043 Q17[31]3.78418
for (int W12t13b = 0; W12t13b < 2; ++W12t13b) { m1[12] ^= 0x2000; m1[14] ^= 0x1800;  
W13[15] w14[12]fwQ15 0.830078 fwQ16 25.5859 Q16[1]0.317383 Q16[29]6.61621 Q16[30]6.61621 Q16[31]6.61621 Q17[28]2.9541
for (int W13t15b = 0; W13t15b < 2; ++W13t15b) { m1[13] ^= 0x8000; m1[14] ^= 0x1000;  
W13[30] w14[12]fwQ16 62.5977 Q16[27]0.12207 Q16[29]0.0488281 Q17[1]0.0732422
for (int W13t30b = 0; W13t30b < 2; ++W13t30b) { m1[13] ^= 0x40000000; m1[14] ^= 0x1000;  
*/
