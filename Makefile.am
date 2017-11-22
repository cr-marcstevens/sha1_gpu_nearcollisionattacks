SUFFIXES=.cu

EXTRA_DIST       = README.md LICENSE.TXT
ACLOCAL_AMFLAGS  = -I m4

AM_CPPFLAGS=-I$(top_srcdir)/lib

noinst_LIBRARIES=lib/libutil.a

EXTRA_PROGRAMS=\
	bin/sha1_freestart76_basesolgen \
	bin/sha1_freestart76_gpuattack \
	bin/sha1_freestart80_basesolgen \
	bin/sha1_freestart80_gpuattack



lib_libutil_a_SOURCES=\
	lib/types.hpp \
	lib/base64.hpp \
	lib/rng.cpp lib/rng.hpp \
	lib/sha1detail.cpp lib/sha1detail.hpp \
	lib/timer.cpp lib/timer.hpp



bin_sha1_freestart76_gpuattack_SOURCES=\
	freestart76/main.cpp freestart76/main.hpp \
	freestart76/cyclicbuffer.hpp \
	freestart76/tables.hpp \
	freestart76/find_basesolution_mbo1.cpp \
	freestart76/cpu_step18-56.cpp \
	freestart76/cuda_step18-56.cu \
	freestart76/sha1benchmark.cu

bin_sha1_freestart76_gpuattack_LDADD=lib/libutil.a $(CUDA_LIBS)

bin_sha1_freestart76_basesolgen_SOURCES=\
	freestart76/main.cpp freestart76/main.hpp \
	freestart76/cyclicbuffer.hpp \
	freestart76/tables.hpp \
	freestart76/find_basesolution_mbo1.cpp \
	freestart76/cpu_step18-56.cpp \
	freestart76/cuda_dummy.cpp

bin_sha1_freestart76_basesolgen_LDADD=lib/libutil.a $(CUDA_LIBS)

freestart76: bin/sha1_freestart76_gpuattack bin/sha1_freestart76_basesolgen



bin_sha1_freestart80_gpuattack_SOURCES=\
	freestart80/main.cpp freestart80/main.hpp \
	freestart80/cyclicbuffer.hpp \
	freestart80/tables.hpp \
	freestart80/find_basesolution_mbo1.cpp \
	freestart80/cpu_step18-60.cpp \
	freestart80/cuda_step18-60.cu \
	freestart80/sha1benchmark.cu

bin_sha1_freestart80_gpuattack_LDADD=lib/libutil.a $(CUDA_LIBS)

bin_sha1_freestart80_basesolgen_SOURCES=\
	freestart80/main.cpp freestart80/main.hpp \
	freestart80/cyclicbuffer.hpp \
	freestart80/tables.hpp \
	freestart80/find_basesolution_mbo1.cpp \
	freestart80/cpu_step18-60.cpp \
	freestart80/cuda_dummy.cpp

bin_sha1_freestart80_basesolgen_LDADD=lib/libutil.a $(CUDA_LIBS)

freestart80: bin/sha1_freestart80_gpuattack bin/sha1_freestart80_basesolgen



#ifeq ($(GENCODE_FLAGS),)
GENCODE_FLAGS=
$(foreach sm,$(CUDA_SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM:=$(lastword $(sort $(CUDA_SMS)))
#ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS+=-gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
#endif
#endif


NVCCFLAGS=-ccbin $(CXX) -O2 --maxrregcount=64 --ptxas-options=-v -Xcompiler -mcmodel=medium

.cu.o:
	$(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) $(GENCODE_FLAGS) $(addprefix -Xcompiler ,$(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS)) -o $@ -c $<
