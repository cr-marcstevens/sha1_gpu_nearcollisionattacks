/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
            (C) 2016 Pierre Karpman, Centrum Wiskunde & Informatica (CWI), Amsterdam.
*****/

#include "types.hpp"

/// Neutral bits positions //////////////
/// W11: ........ ........ .......x x.......
/// W12: ........ ........ ..xxxxxx xxx.....
/// W13: .x...... ........ x.x.xxxx xxx.....
/// W14: ........ ........ .....xxx xx.x....
/// W15: ........ ........ ...x.xxx xxx.....
#define W11NBALLM 0x00000180
#define W12NBALLM 0x00003FE0
#define W13NBALLM 0x4000AFE0
#define W14NBALLM 0x000007D0
#define W15NBALLM 0x000017E0
/// Flip positions //////////////////////
/// W14: ........ ........ ...FF... ........
#define W14FLALLM 0x00001800


#define Q07BOOMS  0x00000140
#define Q10BOOMS  0x00000080

/// Per step neutral bits masks /////////
/// Q16
#define W11NBQ16M 0x00000180 // ........ ........ .......x x....... (2 of 'em)
#define W12NBQ16M 0x00003E00 // ........ ........ ..xxxxx. ........ (5 of 'em)
#define W13NBQ16M 0x40008000 // .x...... ........ x....... ........ (2 of 'em)
#define W14FLQ16M 0xFFFFE7FF // ........ ........ ...FF... ........ <2 FLIPS>
/// Q17
#define W12NBQ17M 0x000001E0 // ........ ........ .......x xxx..... (4 of 'em)
#define W13NBQ17M 0x00002C00 // ........ ........ ..x.xx.. ........ (3 of 'em)
/// Q18
#define W13NBQ18M 0x000003E0 // ........ ........ ......xx xxx..... (5 of 'em)
#define W14NBQ18M 0x00000400 // ........ ........ .....x.. ........ (1 of 'em)
#define W14FLQ18M 0xFFFFF7FF // ........ ........ ....F... ........ <1 FLIP >
/// Q19
//#define W14NBQ19M 0x000003D0 // ........ ........ ......xx xx.x.... (5 of 'em)
#define W14NBQ19M 0x000002C0 // ........ ........ ......x. xx...... (3 of 'em)
//#define W15NBQ19M 0x00001600 // ........ ........ ...x.xx. ........ (3 of 'em)
#define W15NBQ19M 0x00001400 // ........ ........ ...x.x.. ........ (2 of 'em)
/// Q20
//#define W15NBQ20M 0x000001E0 // ........ ........ .......x xxx..... (4 of 'em)
#define W14NBQ20M 0x00000110 // ........ ........ .......x ...x.... (2 of 'em)
#define W15NBQ20M 0x000003E0 // ........ ........ ......xx xxx..... (5 of 'em)

/// Neutral bits packing (Q16--17)
/// W11: ........ ........ .......x x.......
/// W12: ........ ........ ..xxxxxx xxx.....
/// W13: .x...... ........ x.x.xx.. ........
/// W14: ........ ........ ...xx... ........
///
/// W11:1 W12:2 W13:3 W14:4 q15sol:i
/// .3...442222222223.3.33.11.......
/// iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii

FUNC_PREFIX inline uint32_t unpack_q15idx(uint32_t v0, uint32_t v1) { return v1; }
FUNC_PREFIX inline uint32_t unpack_w11_nbs(uint32_t v0, uint32_t v1) { return v0  & W11NBALLM; }
FUNC_PREFIX inline uint32_t unpack_w12_nbs(uint32_t v0, uint32_t v1) { return (v0 >> 11)  & W12NBALLM; }
FUNC_PREFIX inline uint32_t unpack_w13_nbs(uint32_t v0, uint32_t v1) { return v0 & (W13NBQ16M|W13NBQ17M); }
FUNC_PREFIX inline uint32_t unpack_w14_fls(uint32_t v0, uint32_t v1) { return (v0 >> 14) & W14FLALLM; }

FUNC_PREFIX inline uint32_t pack_q16q17_sol0(uint32_t q15solidx, uint32_t w11, uint32_t w12, uint32_t w13, uint32_t w14) { return (w11&W11NBALLM) ^ ((w12&W12NBALLM) << 11) ^ (w13&W13NBALLM) ^ ((w14 & W14FLALLM) << 14); }
FUNC_PREFIX inline uint32_t pack_q16q17_sol1(uint32_t q15solidx, uint32_t w11, uint32_t w12, uint32_t w13, uint32_t w14) { return q15solidx; }

/// Neutral bits packing (Q19--Q21)
/// W14: ........ ........ ......xx xx.x....
/// W15: ........ ........ ...x.xxx xxx.....
///
/// W14:4 W15:5 q18sol:i
/// 444454555555iiiiiiiiiiiiiiiiiiii

FUNC_PREFIX inline uint32_t unpack_q18idx(uint32_t v0) { return (v0 & 0xFFFFF); }
FUNC_PREFIX inline uint32_t unpack_w14_nbs(uint32_t v0) { return (v0 >> 22) & W14NBALLM; } // WARNING: this would be incorrect if the last NB wasn't left-dropped (same below)
FUNC_PREFIX inline uint32_t unpack_w15_nbs(uint32_t v0) { return (v0 >> 15) & W15NBALLM; }

FUNC_PREFIX inline uint32_t pack_q19q21_sol0(uint32_t q18solidx, uint32_t w14, uint32_t w15) { return ((w14&W14NBALLM) << 22) ^ ((w15&W15NBALLM) << 15) ^ (q18solidx & 0xFFFFF); }
