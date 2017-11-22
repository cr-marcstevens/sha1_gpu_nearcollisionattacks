/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
  
  This file is part of sha1freestart80 source-code and released under the MIT License
*****/
    
#ifndef HASHCLASH_TIMER_HPP
#define HASHCLASH_TIMER_HPP

#include "types.hpp"

namespace hc
{

	class timer_detail;
	class timer {
	public:
		timer(bool direct_start = false);
		~timer();
		void start();
		void stop();
		double time() const;// get time between start and stop (or now if still running) in seconds
		bool isrunning() const { return running; } // check if timer is running

	private:
		timer_detail* detail;
		bool running;
	};

} // namespace hash

#endif // HASHCLASH_TIMER_HPP
