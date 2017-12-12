#
#  Copyright (C) 2015 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
#            (C) 2015 Pierre Karpman, INRIA France/Nanyang Technological University Singapore (-2016), CWI (2016/2017), L'Universite Grenoble Alpes (2017-)
#
#  This file is part of sha1_gpu_nearcollisionattacks source-code and released under the MIT License
#

SUFFIXES=.cu

EXTRA_DIST       = README.md LICENSE.TXT
ACLOCAL_AMFLAGS  = -I m4

AM_CPPFLAGS=-I$(top_srcdir)/lib -I$(top_srcdir)/contrib

noinst_LIBRARIES=lib/libutil.a

noinst_PROGRAMS=\
	bin/freestart76_basesolgen \
	bin/freestart76_gpuattack \
	bin/freestart80_basesolgen \
	bin/freestart80_gpuattack \
	bin/shatterednc2_basesolgen \
	bin/shatterednc2_gpuattack



lib_libutil_a_SOURCES=\
	lib/cuda_cyclicbuffer.hpp \
	contrib/types.hpp \
	contrib/rng.cpp contrib/rng.hpp \
	contrib/sha1detail.cpp contrib/sha1detail.hpp \
	contrib/program_options.hpp \
	contrib/base64.hpp \
	contrib/timer.hpp



bin_freestart76_gpuattack_SOURCES=\
	freestart76/main.cpp freestart76/main.hpp \
	freestart76/tables.hpp \
	freestart76/find_basesols.cpp \
	freestart76/cpu_step18-56.cpp \
	freestart76/cuda_step18-56.cu \
	lib/sha1benchmark.cu

bin_freestart76_gpuattack_LDADD=lib/libutil.a $(CUDA_LIBS)

bin_freestart76_basesolgen_SOURCES=\
	freestart76/main.cpp freestart76/main.hpp \
	freestart76/tables.hpp \
	freestart76/find_basesols.cpp \
	freestart76/cpu_step18-56.cpp \
	lib/cuda_dummy.cpp

bin_freestart76_basesolgen_LDADD=lib/libutil.a

freestart76: bin/freestart76_gpuattack bin/freestart76_basesolgen



bin_freestart80_gpuattack_SOURCES=\
	freestart80/main.cpp freestart80/main.hpp \
	freestart80/tables.hpp \
	freestart80/neutral_bits_packing.hpp \
	freestart80/find_basesols.cpp \
	freestart80/cpu_step18-60.cpp \
	freestart80/cuda_step18-60.cu \
	lib/sha1benchmark.cu

bin_freestart80_gpuattack_LDADD=lib/libutil.a $(CUDA_LIBS)

bin_freestart80_basesolgen_SOURCES=\
	freestart80/main.cpp freestart80/main.hpp \
	freestart80/tables.hpp \
	freestart80/neutral_bits_packing.hpp \
	freestart80/find_basesols.cpp \
	freestart80/cpu_step18-60.cpp \
	lib/cuda_dummy.cpp

bin_freestart80_basesolgen_LDADD=lib/libutil.a

freestart80: bin/freestart80_gpuattack bin/freestart80_basesolgen



bin_shatterednc2_gpuattack_SOURCES=\
	shattered_nc2/main.cpp shattered_nc2/main.hpp \
	shattered_nc2/tables_org.hpp \
	shattered_nc2/tables_wQ14nb.hpp \
	shattered_nc2/tables_wQ17nb.hpp \
	shattered_nc2/tables_wboom.hpp \
	shattered_nc2/neutral_bits_packing.hpp \
	shattered_nc2/find_basesols.cpp \
	shattered_nc2/cuda_step14-60.cu \
	lib/sha1benchmark.cu

bin_shatterednc2_gpuattack_LDADD=lib/libutil.a $(CUDA_LIBS)

bin_shatterednc2_basesolgen_SOURCES=\
	shattered_nc2/main.cpp shattered_nc2/main.hpp \
	shattered_nc2/tables_org.hpp \
	shattered_nc2/tables_wQ14nb.hpp \
	shattered_nc2/tables_wQ17nb.hpp \
	shattered_nc2/tables_wboom.hpp \
	shattered_nc2/neutral_bits_packing.hpp \
	shattered_nc2/find_basesols.cpp \
	lib/cuda_dummy.cpp

bin_shatterednc2_basesolgen_LDADD=lib/libutil.a

shatterednc2: bin/shatterednc2_gpuattack bin/shatterednc2_basesolgen


#### CUDA Makefile rules ####
GENCODE_FLAGS=
$(foreach sm,$(CUDA_SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM:=$(lastword $(sort $(CUDA_SMS)))
GENCODE_FLAGS+=-gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

NVCCFLAGS=-ccbin $(CXX) -O2 --maxrregcount=64 --ptxas-options=-v -Xcompiler -mcmodel=medium

.cu.o:
	$(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) $(GENCODE_FLAGS) $(addprefix -Xcompiler ,$(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS)) -o $@ -c $<
