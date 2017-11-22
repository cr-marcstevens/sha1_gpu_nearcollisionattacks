/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

#define HASHCLASH_BUILD

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include "timer.hpp"

namespace hc
{

	class timer_detail {
	public:
#ifdef _WIN32
		LARGE_INTEGER tstart, tend;
		double freq;
#else
		struct timeval tstart, tend;
		struct timezone tz;
#endif
	};

	timer::~timer()
	{
		delete detail;
	}

	timer::timer(bool direct_start): running(false) 
	{ 
		detail = new timer_detail;
#ifdef _WIN32
		LARGE_INTEGER tmp_freq;
		QueryPerformanceFrequency(&tmp_freq);
		detail->freq = double(tmp_freq.QuadPart);
#endif
		if (direct_start)
			start();
	}

#ifdef _WIN32

	void timer::start()
	{
		running = true;
		QueryPerformanceCounter(&detail->tstart);
	}

	void timer::stop()
	{
		QueryPerformanceCounter(&detail->tend);
		running = false;
	}

	double timer::time() const
	{
		if (running)
		{
			LARGE_INTEGER tmp_end;
			QueryPerformanceCounter(&tmp_end);
			return (double(tmp_end.QuadPart) - double(detail->tstart.QuadPart))/detail->freq;
		} else 
			return (double(detail->tend.QuadPart) - double(detail->tstart.QuadPart))/detail->freq;
	}

#else

	void timer::start()
	{
		running = true;
		gettimeofday(&detail->tstart, &detail->tz);
	}

	void timer::stop()
	{
		gettimeofday(&detail->tend, &detail->tz);
		running = false;
	}

	double timer::time() const
	{
		double t1 = double(detail->tstart.tv_sec) + (double(detail->tstart.tv_usec)/1e6);
		if (running)
		{
			struct timeval tmp_end;
			gettimeofday(&tmp_end, &detail->tz);
			return double(tmp_end.tv_sec) + (double(tmp_end.tv_usec)/1e6) - t1;
		} else
			return double(detail->tend.tv_sec) + (double(detail->tend.tv_usec)/1e6) - t1;
	}

#endif

} //namespace hash
