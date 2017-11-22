/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
            (C) 2016 Pierre Karpman

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

#include "types.hpp"

/// Neutral bits positions //////////////
/// W14: ........ ........ ....xxxx x.x.....
/// W15: ........ .......x xxxxxxxx x..x....
/// W16: ........ .......x xxxxxxxx ........
/// W17: ........ ....xxxx xxxxxx.. ........
/// W18: ........ ........ xxxxxxxx xx.x....
/// W19: ........ ........ .x.xxxxx xx......
/// W20: ........ ........ x.xxx... .x......
#define W14NBALLM 0x00000FA0
#define W15NBALLM 0x0001FF90
#define W16NBALLM 0x0001FF00
#define W17NBALLM 0x000FFC00
#define W18NBALLM 0x0000FFD0
#define W19NBALLM 0x00005FC0
#define W20NBALLM 0x0000B840

#define Q12BOOMS  0x00000300
#define Q11BOOMS  0x00000180


/// Neutral bits positions //////////////
/// W17 ALL : ........ ....xxxx xxxxxx.. ........
/// W17 BASE: ........ ....xxxx xx...... ........
/// W17 EXT : ........ ........ ..xxxx.. ........
#define W17NBBASM 0x000FC000
#define W17NBEXTM 0x00003C00
/// Flips
/// F19: ........ ........ x....... ........
#define W19FBM 0x00008000

/// Per step neutral bits masks /////////
/// Q18
#define W14NBQ18M 0x00000F00 // ........ ........ ....xxxx ........ (4 of 'em)
#define W15NBQ18M 0x0001E000 // ........ .......x xxx..... ........ (4 of 'em)
/// Q19
#define W14NBQ19M 0x000000A0 // ........ ........ ........ x.x..... (2 of 'em)
#define W15NBQ19M 0x00001F00 // ........ ........ ...xxxNN ........ (5 of 'em)
#define W16NBQ19M 0x0001F000 // ........ .......x xNNN.... ........ (5 of 'em)
/// Q20
#define W15NBQ20M 0x00000090 // ........ ........ ......OO x..x.... (2 of 'em)
#define W16NBQ20M 0x00000F00 // ........ ........ .OOOxxxx ........ (4 of 'em)
#define W17NBQ20M 0x000FC000 // ........ ....xxxx xx...... ........ (6 of 'em)
/// Q21
#define W17NBQ21M 0x00003C00 // ........ ........ ..xxxx.. ........ (4 of 'em)
#define W18NBQ21M 0x00008000 // ........ ........ x....... ........ (1 of 'em)
/// Q22
#define W18NBQ22M 0x00007E00 // ........ ........ .xxxxxx. ........ (6 of 'em)
#define W19NBQ22M 0x00004400 // ........ ........ .x...x.. ........ (2 of 'em)
/// Q23
#define W18NBQ23M 0x000001D0 // ........ ........ .......x xx.x.... (4 of 'em)
#define W19NBQ23M 0x00001A00 // ........ ........ ...xx.x. ........ (3 of 'em)
#define W20NBQ23M 0x00008000 // ........ ........ x....... ........ (1 of 'em)
/// Q24
#define W19NBQ24M 0x000001C0 // ........ ........ .......x xx...... (3 of 'em)
#define W20NBQ24M 0x00003800 // ........ ........ ..xxx... ........ (3 of 'em)
/// Q25
#define W20NBQ25M 0x00000040 // ........ ........ ........ .x...... (1 of 'em)

/// Neutral bits packing (Q18--20) 
///
/// W14: ........ ........ ....xxxx x.x.....
/// W15: ........ .......x xxxxxxxx x..x....
/// W16: ........ .......x xxxxxxxx ........
/// W17: ........ ....xxxx xx...... ........
///
/// W14:a W15:b W16:c W17:d basesol:i
/// ccccccccc...bbbbbbbbbb..baaaaa.a
/// dddddd......iiiiiiiiiiiiiiiiiiii

FUNC_PREFIX inline uint32_t unpack_idx(uint32_t v0, uint32_t v1) { return v1 & 0xFFFFF; }
FUNC_PREFIX inline uint32_t unpack_w14_nbs(uint32_t v0, uint32_t v1) { return (v0 << 5) & W14NBALLM; }
FUNC_PREFIX inline uint32_t unpack_w15_nbs(uint32_t v0, uint32_t v1) { return (v0 >> 3) & W15NBALLM; }
FUNC_PREFIX inline uint32_t unpack_w16_nbs(uint32_t v0, uint32_t v1) { return (v0 >> 15) & W16NBALLM; }
FUNC_PREFIX inline uint32_t unpack_w17base_nbs(uint32_t v0, uint32_t v1) { return (v1 >> 12) & W17NBBASM; }

FUNC_PREFIX inline uint32_t pack_q18q20_sol0(uint32_t basesolidx, uint32_t w14, uint32_t w15, uint32_t w16 = 0, uint32_t w17 = 0) { return ((w14&W14NBALLM) >> 5) ^ ((w15&W15NBALLM) << 3) ^ ((w16&W16NBALLM) << 15); }
FUNC_PREFIX inline uint32_t pack_q18q20_sol1(uint32_t basesolidx, uint32_t w14, uint32_t w15, uint32_t w16 = 0, uint32_t w17 = 0) { return ((w17&W17NBBASM) << 12) ^ (basesolidx & 0xFFFFF); }

/// Neutral bits packing (Q21--Q25)
/// W17: ........ ........ ..xxxx.. ........
/// W18: ........ ........ xxxxxxxx xx.x....
/// W19: ........ ......F. Fx.xxxxx xx......
/// W20: ........ ......F. x.xxx... .x.....F
///
/// W17:a W18:b W19:c W20:d extsol:i
/// bbbbbbbbbb.b..F.d.ddd....d.aaaaF
/// F.Fc.ccccccciiiiiiiiiiiiiiiiiiii

#define W19NBPACKM 0x0002DFC0
#define W20NBPACKM 0x0002B841

FUNC_PREFIX inline uint32_t unpack_w17ext_nbs(uint32_t v0, uint32_t v1) { return (v0 << 9) & W17NBEXTM; }
FUNC_PREFIX inline uint32_t unpack_w18_nbs(uint32_t v0, uint32_t v1) { return (v0 >> 16) & W18NBALLM; }
FUNC_PREFIX inline uint32_t unpack_w19_nbs(uint32_t v0, uint32_t v1) { return (v1 >> 14) & W19NBALLM; }
FUNC_PREFIX inline uint32_t unpack_w20_nbs(uint32_t v0, uint32_t v1) { return v0 & W20NBALLM; }
FUNC_PREFIX inline uint32_t unpack_w19_nbs_fb(uint32_t v0, uint32_t v1) { return (v1 >> 14) & W19NBPACKM; }
FUNC_PREFIX inline uint32_t unpack_w20_nbs_fb(uint32_t v0, uint32_t v1) { return v0 & W20NBPACKM; }

FUNC_PREFIX inline uint32_t pack_q21q25_sol0(uint32_t extsolidx, uint32_t w17, uint32_t w18, uint32_t w19 = 0, uint32_t w20 = 0) { return ((w18&W18NBALLM) << 16) ^ ((w17&W17NBEXTM) >> 9) ^ (w20 & W20NBPACKM); }
FUNC_PREFIX inline uint32_t pack_q21q25_sol1(uint32_t extsolidx, uint32_t w17, uint32_t w18, uint32_t w19 = 0, uint32_t w20 = 0) { return ((w19&W19NBPACKM) << 14) ^ (extsolidx & 0xFFFFF); }
