# SHA-1 GPU near-collision attacks

## Publications

This repository contains the source code belonging to three scientific publications:

- [*Practical free-start collision attacks on 76-step SHA-1*](https://eprint.iacr.org/2015/530), Pierre Karpman, Thomas Peyrin, and Marc Stevens, CRYPTO 2015, Lecture Notes in Computer Science, vol. 9215, Springer, 2015, pp. 623-642.

  This publication introduces the efficient GPU framework for SHA-1 collision finding, improvements of cryptanalytic tools, and a freestart attack on 76-step SHA-1. The runtime cost is roughly 5 days on 1 NVIDIA GTX-970 GPU and was executed on the GPU cluster of Thomas Peyrin.

- [*Freestart collision for full SHA-1*](https://eprint.iacr.org/2015/967), Marc Stevens, Pierre Karpman, Thomas Peyrin, EUROCRYPT 2016, Lecture Notes in Computer Science, vol. 9665, Springer, 2016, pp. 459-483.

  This publication introduces a freestart attack on 80-step SHA-1, further improvements of cryptanalytic tools, and presents a cost analysis for a (normal) collision attack for full SHA-1. The runtime cost is roughly 640 days on 1 NVIDIA GTX-970 GPU and was executed on the GPU cluster of Thomas Peyrin.
  
  See also https://sites.google.com/site/itstheshappening/

- [*The first collision for full SHA-1*](https://eprint.iacr.org/2017/190), Marc Stevens, Elie Bursztein, Pierre Karpman, Ange Albertini, Yarik Markov, CRYPTO 2017, Lecture Notes in Computer Science, vol. 10401, Springer, 2017, pp. 570-596.

  This publication presents a complete collision attack on full SHA-1 and further improvements of cryptanalytic tools. The majority of the runtime cost is roughly 102 years on 1 NVIDIA GTX-970 GPU and was executed on a distributed GPU system of Google. 
  Note that only the second near-collision attack was implemented for GPU and is released here without the changes to adapt it to Google's proprietary build and job systems. 
  The first near-collision attack of this project was already published in EUROCRYPT 2013 and is available at https://github.com/cr-marcstevens/hashclash.

  See also https://shattered.it

## Requirements

 - CUDA SDK

 - C++11 compiler compatible with CUDA SDK

 - autotools

## Building

 - `autoreconf --install`

 - `./configure [--with-cuda=/usr/local/cuda-X.X] [--enable-cudagencode=50,52]`

 - `make`

## Find your own 76-round SHA-1 freestart collision

 - Expected GPU runtime: *5 days* on a single GTX-970

 - `mkdir fs76; cd fs76`

 - `../run_freestart76.sh`

 - Example manual usage tools:
 
   - `bin/freestart76_basesolgen --genbasesol --seed 4_23_152443400808031284 --maxbasesols 262144 -o fs76_basesols.bin`
   
   - `bin/freestart76_gpuattack --cudaattack -i fs76_basesols.bin -o fs76_q56sols.bin`
   
   - `bin/freestart76_basesolgen --verifyQ56 -i fs76_q56sols.bin | grep Found -B88 -A52`

## Find your own 80-round SHA-1 freestart collision

 - Expected GPU runtime: *640 days* on a single GTX-970

 - Generate basesolutions (32 base64-encoded per textline)

 `bin/freestart80_basesolgen -g -o basesol80_1.txt -m 1024`

 - Run GPU attack:

 `bin/freestart80_gpuattack -a -i basesol80_1.txt`

## Find your own shattered 2nd near-collision block pair

 - Expected GPU runtime: *102 years* on a single GTX-970

 - Generate basesolutions (32 base64-encoded per textline)

 `bin/shatterednc2_basesolgen -g -o basesolnc2_1.txt -m 1024`

 - Run GPU attack:

 `bin/shatterednc2_gpuattack -a -i basesolnc2_1.txt`
