# sha1_gpu_nearcollisionattacks

## Requirements

- CUDA SDK

- C++11 compiler compatible with CUDA SDK

- autotools

# Building

- `autoreconf --install`

- `./configure [--with-cuda=/usr/local/cuda-X.X] [--enable-cudagencode=50,52]`

- `make freestart76`

# Find your own 76-round SHA-1 freestart collision

- `mkdir fs76; cd fs76`

- `../run_freestart76.sh`
