AC_PREREQ([2.68])
AC_INIT([SHA1_GPU_NCA], [1.0.0], [marc@marc-stevens.nl],[SHA1_GPU_NCA],[https://github.com/cr-marcstevens/sha1_gpu_nearcollisionattacks])
AC_CONFIG_SRCDIR([README.md])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign subdir-objects])


DEFAULT_CXXFLAGS="-O2 -Wfatal-errors"
dnl DEFAULT_CXXFLAGS="-ggdb -g"
AS_IF([test "x$CXXFLAGS" = "x"],
	[CXXFLAGS="$DEFAULT_CXXFLAGS"]
	[usedefaultcxxflags=yes],
	[usedefaultcxxflags=no])

LT_INIT

AC_LANG([C++])

AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_LN_S
AC_PROG_MKDIR_P
AC_PROG_GREP
AC_PROG_CXX

AX_CXX_COMPILE_STDCXX([11],[noext],[mandatory])

AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

AS_IF([test "x$cross_compiling" != "xyes" && test "x$usedefaultcxxflags" = "xyes" ],
	[AX_CHECK_COMPILE_FLAG([-march=native], [CXXFLAGS="$CXXFLAGS -march=native"], [])])


AX_PTHREAD()
LIBS="$PTHREAD_LIBS $LIBS"
CXXFLAGS="$CXXFLAGS $PTHREAD_CFLAGS"


AX_CUDA
AM_COND_IF([HAVE_CUDA],AC_MSG_NOTICE([Cuda found]),AC_MSG_ERROR([Cuda is required!]))


AC_ARG_ENABLE([cudagencode], AS_HELP_STRING([--enable-cudagencode@<:@..,50,52,..@:>@],[Compile for specific cuda compute capabilities (default: 50,52)])
	,[cudasms="$enableval"],[cudasms="50,52"])
CUDA_SMS=`echo "$cudasms" | tr -c 0123456789 ' '`
AS_IF([test "$CUDA_SMS" = "" ],[AC_MSG_ERROR([invalid list of cuda compute capabilities to compile to: $CUDA_SMS])])
AC_SUBST(CUDA_SMS)


AC_CONFIG_FILES([
 Makefile
])


AC_OUTPUT
