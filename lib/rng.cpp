/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/


#define HASHCLASH_BUILD

#include <time.h>
#include <iostream>
#include "rng.hpp"

#if defined(__linux__) || defined (__FreeBSD__)
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>    // open
#include <unistd.h>   // read, close
	void getosrnd(uint32_t buf[256])
	{
		int fd;
		if ((fd = open("/dev/urandom", O_RDONLY)) < 0) return;
		// ignore any warnings from this read
		read(fd, reinterpret_cast<char*>(buf), 256);
		close(fd);
	}
#endif

#if defined(WIN32)
#include <windows.h>
#include <wincrypt.h>
	void getosrnd(uint32_t buf[256])
	{
		HCRYPTPROV g_hCrypt;
		if(!CryptAcquireContext(&g_hCrypt,NULL,NULL,PROV_RSA_FULL,CRYPT_VERIFYCONTEXT))
			return;
		CryptGenRandom(g_hCrypt,sizeof(buf),reinterpret_cast<BYTE*>(buf));
		CryptReleaseContext(g_hCrypt,0);
	}
#endif


	void getosrnd(uint32_t buf[256]);

	uint32_t seedd;
	uint32_t seed32_1;
	uint32_t seed32_2;
	uint32_t seed32_3;
	uint32_t seed32_4;

	void seed(uint32_t s)
	{
		seedd = 0;
		seed32_1 = s;
		seed32_2 = 2;
		seed32_3 = 3;
		seed32_4 = 4;
		for (unsigned i = 0; i < 0x1000; ++i)
			xrng128();
	}

	void seed(uint32_t* sbuf, unsigned len)
	{
		seedd = 0;
		seed32_1 = 1;
		seed32_2 = 2;
		seed32_3 = 3;
		seed32_4 = 4;
		for (unsigned i = 0; i < len; ++i)
		{
			seed32_1 ^= sbuf[i];
			xrng128();
		}
		for (unsigned i = 0; i < 0x1000; ++i)
			xrng128();
	}

	void addseed(uint32_t s)
	{
		xrng128();
		seed32_1 ^= s;
		xrng128();
	}

	void addseed(const uint32_t* sbuf, unsigned len)
	{
		xrng128();
		for (unsigned i = 0; i < len; ++i)
		{
			seed32_1 ^= sbuf[i];
			xrng128();
		}
	}

	struct hashutil5_rng__init {
		hashutil5_rng__init()
		{
			addseed(uint32_t(time(NULL)));
			uint32_t rndbuf[256];
			for (unsigned i = 0; i < 256; ++i) rndbuf[i] = 0;
			getosrnd(rndbuf);
			addseed(rndbuf, 256);
		}
	};
	hashutil5_rng__init hashutil4_rng__init__now;

	void hashutil5_rng_hpp_init()
	{
		hashutil5_rng__init here;
	}

