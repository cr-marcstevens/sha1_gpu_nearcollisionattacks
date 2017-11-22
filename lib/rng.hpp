/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

/* 

very fast inlined xorshift random number generators with periods 2^32 - 1, 2^64 - 1, 2^96 - 1 and 2^128 - 1
by G. Marsaglia: http://www.jstatsoft.org/v08/i14/xorshift.pdf 
this implementation only uses time to seed!

*/

#ifndef HASHCLASH_XORSHIFT_RNG_HPP
#define HASHCLASH_XORSHIFT_RNG_HPP

#include <stdint.h>

	// seed all generators using 32-bit values
	// seed state is deterministically dependent on the given values
	void seed(uint32_t s);
	void seed(const uint32_t* sbuf, unsigned len);
	// add seed to the generators
	// seed state is changed by given values
	void addseed(uint32_t s);
	void addseed(const uint32_t* sbuf, unsigned len);

	// seeds used, these are initialized to random values based on the time
	extern uint32_t seedd;
	extern uint32_t seed32_1;
	extern uint32_t seed32_2;
	extern uint32_t seed32_3;
	extern uint32_t seed32_4;

	/******** Random generator with perdiod (2^32 - 1)*2^32 **********/
	inline uint32_t xrng32()
	{
		seed32_1 ^= seed32_1 << 13;
		seed32_1 ^= seed32_1 >> 17;
		return (seed32_1 ^= seed32_1 << 5) + (seedd += 789456123);
	}

	/******** Random generator with perdiod (2^64 - 1)*2^32 **********/
	inline uint32_t xrng64()
	{
		uint32_t t = seed32_1 ^ (seed32_1 << 10);
		seed32_1 = seed32_2;
		seed32_2 = (seed32_2^(seed32_2>>10))^(t^(t>>13));
		return seed32_1 + (seedd += 789456123);
	}

	/******** Random generator with perdiod (2^96 - 1)*2^32 **********/
	inline uint32_t xrng96()
	{
		uint32_t t = seed32_1 ^ (seed32_1 << 10);
		seed32_1 = seed32_2;
		seed32_2 = seed32_3;
		seed32_3 = (seed32_3^(seed32_3>>26))^(t^(t>>5));
		return seed32_1 + (seedd += 789456123);
	}

	/******** Random generator with perdiod (2^128 - 1)*2^32 **********/
	inline uint32_t xrng128()
	{
		uint32_t t = seed32_1 ^ (seed32_1 << 5);
		seed32_1 = seed32_2;
		seed32_2 = seed32_3;
		seed32_3 = seed32_4;
		seed32_4 = (seed32_4^(seed32_4>>1))^(t^(t>>14));
		return seed32_1 + (seedd += 789456123);
	}

	class xrng {
	public:
		xrng() { seed(); }
		xrng(xrng& r) { seed(r); }
		xrng& operator=(xrng& r) { seed(r); return *this; }
		
		void seed() {
			_seed32_1 = ::xrng128(); 
			_seed32_2 = ::xrng128();
			_seed32_3 = ::xrng128();
			_seed32_4 = ::xrng128();
			_seedd = ::xrng128();
		}
		void seed(xrng& r)
		{
			_seed32_1 = r.xrng128(); 
			_seed32_2 = r.xrng128();
			_seed32_3 = r.xrng128();
			_seed32_4 = r.xrng128();
			_seedd = r.xrng128();
		}
		void addseed(const uint32_t s)
		{
			_seed32_1 ^= s;
			this->xrng128();			
			this->xrng128();			
			this->xrng128();			
			this->xrng128();			
			this->xrng128();			
		}
		void addseed(const uint32_t s[], const unsigned len)
		{
			for (unsigned i = 0; i < len; ++i)
			{
				_seed32_1 ^= s[i];
				this->xrng128();
			}
			this->xrng128();			
			this->xrng128();			
			this->xrng128();			
			this->xrng128();			
		}
		template<class T>
		void addseed(const T& s)
		{
			const unsigned char* ptr = reinterpret_cast<const unsigned char*>(&s);
			for (unsigned i = 0; i < sizeof(T); ++i)
			{
				_seed32_1 ^= uint32_t(ptr[i]);
				this->xrng128();			
			}
			this->xrng128();			
			this->xrng128();			
			this->xrng128();			
			this->xrng128();			
		}
		/******** Random generator with perdiod (2^32 - 1)*2^32 **********/
		uint32_t xrng32()
		{
			_seed32_1 ^= _seed32_1 << 13;
			_seed32_1 ^= _seed32_1 >> 17;
			return (_seed32_1 ^= _seed32_1 << 5) + (_seedd += 789456123);
		}

		/******** Random generator with perdiod (2^64 - 1)*2^32 **********/
		uint32_t xrng64()
		{
			uint32_t t = _seed32_1 ^ (_seed32_1 << 10);
			_seed32_1 = _seed32_2;
			_seed32_2 = (_seed32_2^(_seed32_2>>10))^(t^(t>>13));
			return _seed32_1 + (_seedd += 789456123);
		}

		/******** Random generator with perdiod (2^96 - 1)*2^32 **********/
		uint32_t xrng96()
		{
			uint32_t t = _seed32_1 ^ (_seed32_1 << 10);
			_seed32_1 = _seed32_2;
			_seed32_2 = _seed32_3;
			_seed32_3 = (_seed32_3^(_seed32_3>>26))^(t^(t>>5));
			return _seed32_1 + (_seedd += 789456123);
		}

		/******** Random generator with perdiod (2^128 - 1)*2^32 **********/
		uint32_t xrng128()
		{
			uint32_t t = _seed32_1 ^ (_seed32_1 << 5);
			_seed32_1 = _seed32_2;
			_seed32_2 = _seed32_3;
			_seed32_3 = _seed32_4;
			_seed32_4 = (_seed32_4^(_seed32_4>>1))^(t^(t>>14));
			return _seed32_1 + (_seedd += 789456123);
		}
	private:
		uint32_t _seed32_1, _seed32_2, _seed32_3, _seed32_4, _seedd;
	};

	// call this in a other global constructor using any xrng (without seeding itself)
	// due to the inpredictable order of which global constructors are called
	// otherwise the correct seeding of xrng is not guaranteed
	void hashutil5_rng_hpp_init();

#endif // HASHCLASH_XORSHIFT_RNG_HPP

