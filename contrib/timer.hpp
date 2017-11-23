/*********************************************************************************\
*                                                                                 *
* https://github.com/cr-marcstevens/snippets/tree/master/cxxheaderonly            *
*                                                                                 *
* timer.hpp - A header only C++ simple timer                                      *
* Copyright (c) 2017 Marc Stevens                                                 *
*                                                                                 *
* MIT License                                                                     *
*                                                                                 *
* Permission is hereby granted, free of charge, to any person obtaining a copy    *
* of this software and associated documentation files (the "Software"), to deal   *
* in the Software without restriction, including without limitation the rights    *
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       *
* copies of the Software, and to permit persons to whom the Software is           *
* furnished to do so, subject to the following conditions:                        *
*                                                                                 *
* The above copyright notice and this permission notice shall be included in all  *
* copies or substantial portions of the Software.                                 *
*                                                                                 *
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      *
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     *
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          *
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   *
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   *
* SOFTWARE.                                                                       *
*                                                                                 *
\*********************************************************************************/

#ifndef SNIPPETS_TIMER_HPP
#define SNIPPETS_TIMER_HPP

#include <chrono>

namespace timer
{
	class timer {
	public:
		typedef std::chrono::steady_clock clock_t;
		typedef clock_t::time_point time_point_t;

		timer()
			: _start(clock_t::now()), _running(true)
		{
		}

		inline void start()
		{
			_running = true;
			_start = clock_t::now();
		}

		inline void stop()
		{
			_interval = clock_t::now() - _start;
			_running = false;
		}

		inline double time()
		{
			if (_running)
				_interval = clock_t::now() - _start;
			return _interval.count();
		}

		inline bool isrunning() const { return _running; }

	private:
		time_point_t _start;
		bool _running;
		std::chrono::duration<double> _interval;
	};
}

#endif // SNIPPETS_TIMER_HPP
