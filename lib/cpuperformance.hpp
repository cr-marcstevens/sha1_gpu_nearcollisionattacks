/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

#ifndef HASHCLASH_CPUPERFORMANCE_HPP
#define HASHCLASH_CPUPERFORMANCE_HPP


#include <vector>
#include <string>
#include <iostream>
#include <cstdio>

#ifndef __GNUC__
#include <intrin.h>
#endif

namespace hc {

	inline uint64_t cpu_timestamp()
	{
#ifdef __GNUC__
		uint32_t highpart, lowpart;
		asm volatile("rdtsc": "=d"(highpart), "=a"(lowpart));
		return (uint64_t(highpart) << 32) | uint64_t(lowpart);
#else
		return __rdtsc();
#endif
		
	}

	inline void start_update_counter(uint64_t& performancecounter)
	{
		performancecounter -= cpu_timestamp();
	}
	inline void end_update_counter(uint64_t& performancecounter)
	{
		performancecounter += cpu_timestamp();
	}

	class update_performance_counter {
		uint64_t& _counter;
	public:
		update_performance_counter(uint64_t& performance_counter)
			: _counter(performance_counter)
		{
			_counter -= cpu_timestamp();
		}
		~update_performance_counter()
		{
			_counter += cpu_timestamp();
		}
	};

	class performance_counter_manager {
		std::vector<uint64_t*> counters;
		std::vector<std::string> descriptions;
		uint64_t cputs_start;
	public:
		performance_counter_manager(): cputs_start(cpu_timestamp()) {}

		void add_performance_counter(uint64_t& counter, const std::string& description)
		{
			counters.push_back(&counter);
			descriptions.push_back(description);
		}

		void show_results()
		{
			uint64_t cputime = cpu_timestamp() - cputs_start;
			for (unsigned i = 0; i < counters.size(); ++i)
			{
				std::cout << "Counter " << i << ": \t" << descriptions[i] << std::endl;
				std::cout << "Counter " << i << ": \tValue = " << (*counters[i]) << " \t" << (100.0*float(*counters[i])/float(cputime)) << "%" << std::endl;
			}
		}
	};
} // namespace hash


#endif // HASHCLASH_CPUPERFORMANCE_HPP
