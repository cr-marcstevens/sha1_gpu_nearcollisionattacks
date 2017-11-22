#include "main.hpp"

template<bool b>
struct assert_compile_time_bool
{
};
template<>
struct assert_compile_time_bool<true>
{
	typedef bool compile_time_bool_is_true;
};
#define ASSERT_COMPILE_TIME(b) { typename assert_compile_time_bool< (b) >::compile_time_bool_is_true ___assert = true; if (!___assert) { printf("boooh!"); } }

/***** MASK VERSION *****/

// control logic using mask version
template<size_t N>
class cyclic_buffer_control_mask_t
{
	public:

	static const uint32_t size = N;
	volatile uint32_t write_idx;
	volatile uint32_t read_idx;

	__host__ __device__ inline void reset()
	{
		write_idx = 0;
		read_idx = 0;
	}

	// assumes entire warp calls this function with identical warp_to_write_mask
	// returns per-thread idx to write to
	__device__ inline uint32_t warp_write_idx(uint32_t warp_to_write_mask)
	{
		// warp: determine count and offset
		uint32_t count  = __popc(warp_to_write_mask);
		// thread ZERO has offset 0
		uint32_t offset = count - __popc(warp_to_write_mask >> (threadIdx.x&31));
		uint32_t baseidx = 0;
		// thread ZERO increases write_idx with count and obtains old value in baseidx
		if ((threadIdx.x&31)==0)
		{
			baseidx = atomicAdd((uint32_t*)&write_idx,count);
		}
		// thread ZERO passes baseidx to entire warp
		baseidx = __shfl(baseidx,0);
		return (baseidx + offset);
	}

	// assumes entire warp calls this function with identical inputs
	// if read is safe return per-thread idx to read from, returns 0xFFFFFFFF otherwise
	template<typename safereadval_t>
	__device__ inline uint32_t warp_read_idx(safereadval_t safereadvals[N])
	{
		while (true)
		{
			// obtain warp-wide unique race-free value of read_idx
			uint32_t baseidx = __shfl(read_idx,0);
			uint32_t thrdidx = (baseidx+(threadIdx.x&31));
			// if not safe to read then return false
			if (!__all(0 != safereadvals[ thrdidx % N ] ))
			{
				return 0xFFFFFFFF;
			}
			// thread ZERO tries to increase read_idx with 32, baseidx2==baseidx in case of success, != otherwise
			uint32_t baseidx2;
			if ((threadIdx.x&31)==0)
			{
				baseidx2 = atomicCAS((uint32_t*)&read_idx, baseidx, baseidx+32);
			}
			// thread ZERO passes baseidx2 to entire warp
			baseidx2 = __shfl(baseidx2, 0);
			// if read_idx was successfully increased return (true,baseidx+offset)
			if (baseidx2 == baseidx)
			{
				safereadvals[thrdidx % N] = 0;
				return thrdidx;
			}
			// increase failed due to race, try again
		}
	}
	

	/**** HOST FUNCTIONS ASSUME EXCLUSIVE ACCESS FROM 1 HOST THREAD ****/
	template<typename safereadval_t>
	__host__ inline uint32_t host_read_idx(safereadval_t safereadvals[N])
	{
		uint32_t i = read_idx;
		if (safereadvals[i])
		{
			safereadvals[i] = 0;
			++read_idx;
			return i;
		}
		else
		{
			return 0xFFFFFFFF;
		}
	}
	__host__ inline uint32_t host_write_idx()
	{
		uint32_t i = write_idx;
		++write_idx;
		return i;
	}
};

// control logic using mask version
template<size_t N>
class cyclic_buffer_control_mask_readlock_t
{
	public:

	static const uint32_t size = N;
	volatile uint32_t write_idx;
	volatile uint32_t read_idx;
	volatile int lock;

	__host__ __device__ inline void reset()
	{
		write_idx = 0;
		read_idx = 0;
		lock = 0;
	}

	// assumes entire warp calls this function with identical warp_to_write_mask
	// returns per-thread idx to write to
	__device__ inline uint32_t warp_write_idx(uint32_t warp_to_write_mask)
	{
		// warp: determine count and offset
		uint32_t count  = __popc(warp_to_write_mask);
		// thread ZERO has offset 0
		uint32_t offset = count - __popc(warp_to_write_mask >> (threadIdx.x&31));
		uint32_t baseidx = 0;
		// thread ZERO increases write_idx with count and obtains old value in baseidx
		if ((threadIdx.x&31)==0)
		{
			baseidx = atomicAdd((uint32_t*)&write_idx,count);
		}
		// thread ZERO passes baseidx to entire warp
		baseidx = __shfl(baseidx,0);
		return (baseidx + offset);
	}

	// assumes entire warp calls this function with identical inputs
	// if read is safe return per-thread idx to read from, returns 0xFFFFFFFF otherwise
	template<typename safereadval_t>
	__device__ inline uint32_t warp_read_idx(safereadval_t safereadvals[N])
	{
		int is_locked;
		if ((threadIdx.x&31) == 0)
		{
			is_locked = atomicCAS((int *)&lock, 0, 1);
		}
		is_locked =__shfl(is_locked, 0);
		if (is_locked)
		{
			return 0xFFFFFFFF;
		}
		else	
		{
			// obtain warp-wide unique race-free value of read_idx
			uint32_t baseidx = __shfl(read_idx,0);
			uint32_t thrdidx = (baseidx+(threadIdx.x&31));
			// if not safe to read then return false
			if (!__all(0 != safereadvals[ thrdidx % N ] ))
			{
				if ((threadIdx.x&31) == 0)
				{
					atomicSub((int *)&lock, 1);
				}
				return 0xFFFFFFFF;
			}
			// thread ZERO tries to increase read_idx with 32, baseidx2==baseidx in case of success, != otherwise
			if ((threadIdx.x&31)==0)
			{
				atomicAdd((uint32_t*)&read_idx, 32);
				atomicSub((int *)&lock, 1);
			}
			safereadvals[thrdidx % N] = 0;
			return thrdidx;
		}
	}

	/**** HOST FUNCTIONS ASSUME EXCLUSIVE ACCESS FROM 1 HOST THREAD ****/
	template<typename safereadval_t>
	__host__ inline uint32_t host_read_idx(safereadval_t safereadvals[N])
	{
		uint32_t i = read_idx;
		if (safereadvals[i])
		{
			safereadvals[i] = 0;
			++read_idx;
			return i;
		}
		else
		{
			return 0xFFFFFFFF;
		}
	}
	__host__ inline uint32_t host_write_idx()
	{
		uint32_t i = write_idx;
		++write_idx;
		return i;
	}
};

// fencetype
// - 0 none
// - 1 block-wide
// - 2 gpu-wide
template<int fencetype> __device__ void memoryfence() { __threadfence(); }
template<>              __device__ void memoryfence<0>() {}
template<>              __device__ void memoryfence<1>() { __threadfence_block(); }

template<size_t N, typename val_t = uint32_t, size_t val_cnt = 1, typename control_type = cyclic_buffer_control_mask_t<N>, int fencetype = 2 >
class cyclic_buffer_mask_t
{
	public:

	typedef control_type control_t;

	static const uint32_t size = N;
	static const uint32_t val_size = val_cnt;

	volatile val_t val[val_cnt][N];
	volatile char safe_to_read[N];
	
	// called by entire warp
	__host__ __device__ inline void reset(control_t& control)
	{
#ifdef __CUDA_ARCH__
		for (uint32_t i = (threadIdx.x&31); i < N; i+=32)
			safe_to_read[i] = 0;
#else
		for (uint32_t i = 0; i < N; ++i)
			safe_to_read[i] = 0;
#endif
		control.reset();
	}
	
	// device: called by entire warp
	// host  : assumes exclusive access !!
	__host__ __device__ inline void write(control_t& control, bool dowrite, val_t _val0=0, val_t _val1=0)
	{
#ifdef __CUDA_ARCH__
		uint32_t mask = __ballot(dowrite);
		if (mask == 0) 
		{
			return;
		}
		uint32_t wi = control.warp_write_idx(mask);
		if (dowrite)
		{
			if (0 < val_cnt) val[0][wi % N] = _val0;
			if (1 < val_cnt) val[1][wi % N] = _val1;
//			__threadfence();
			memoryfence<fencetype>();
			safe_to_read[wi % N] = 1;
		}
#else
		if (dowrite)
		{
			uint32_t wi = control.host_write_idx();
			if (0 < val_cnt) val[0][wi % N] = _val0;
			if (1 < val_cnt) val[1][wi % N] = _val1;
			safe_to_read[wi % N] = 1;
		}
#endif
	}

	// device: called by entire warp
	// host  : assumes exclusive access !!
	__host__ __device__ inline void write(control_t& control, bool dowrite
		, val_t _val0   , val_t _val1   , val_t _val2   , val_t _val3 =0, val_t _val4 =0, val_t _val5 =0, val_t _val6 =0, val_t _val7 =0, val_t _val8 =0, val_t _val9 =0
		, val_t _val10=0, val_t _val11=0
		)
	{
#ifdef __CUDA_ARCH__
		uint32_t mask = __ballot(dowrite);
		if (mask == 0) 
		{
			return;
		}
		uint32_t wi = control.warp_write_idx(mask);
		if (dowrite)
		{
#else
		if (dowrite)
		{
			uint32_t wi = control.host_write_idx();
#endif
			if ( 0 < val_cnt) val[ 0][wi % N] = _val0;
			if ( 1 < val_cnt) val[ 1][wi % N] = _val1;
			if ( 2 < val_cnt) val[ 2][wi % N] = _val2;
			if ( 3 < val_cnt) val[ 3][wi % N] = _val3;
			if ( 4 < val_cnt) val[ 4][wi % N] = _val4;
			if ( 5 < val_cnt) val[ 5][wi % N] = _val5;
			if ( 6 < val_cnt) val[ 6][wi % N] = _val6;
			if ( 7 < val_cnt) val[ 7][wi % N] = _val7;
			if ( 8 < val_cnt) val[ 8][wi % N] = _val8;
			if ( 9 < val_cnt) val[ 9][wi % N] = _val9;
			if (10 < val_cnt) val[10][wi % N] = _val10;
			if (11 < val_cnt) val[11][wi % N] = _val11;
#ifdef __CUDA_ARCH__
//			__threadfence();
			memoryfence<fencetype>();
#endif
			safe_to_read[wi % N] = 1;
		}
	}

	// device: called by entire warp
	// host  : assumes exclusive access !!
	__host__ __device__ inline void write(control_t& control, bool dowrite
		, val_t _val0   , val_t _val1   , val_t _val2   , val_t _val3   , val_t _val4   , val_t _val5   , val_t _val6   , val_t _val7   , val_t _val8   , val_t _val9 
		, val_t _val10  , val_t _val11  , val_t _val12  , val_t _val13=0, val_t _val14=0, val_t _val15=0, val_t _val16=0, val_t _val17=0, val_t _val18=0, val_t _val19=0
		, val_t _val20=0, val_t _val21=0
		)
	{
#ifdef __CUDA_ARCH__
		uint32_t mask = __ballot(dowrite);
		if (mask == 0) 
		{
			return;
		}
		uint32_t wi = control.warp_write_idx(mask);
		if (dowrite)
		{
#else
		if (dowrite)
		{
			uint32_t wi = control.host_write_idx();
#endif
			if ( 0 < val_cnt) val[ 0][wi % N] = _val0;
			if ( 1 < val_cnt) val[ 1][wi % N] = _val1;
			if ( 2 < val_cnt) val[ 2][wi % N] = _val2;
			if ( 3 < val_cnt) val[ 3][wi % N] = _val3;
			if ( 4 < val_cnt) val[ 4][wi % N] = _val4;
			if ( 5 < val_cnt) val[ 5][wi % N] = _val5;
			if ( 6 < val_cnt) val[ 6][wi % N] = _val6;
			if ( 7 < val_cnt) val[ 7][wi % N] = _val7;
			if ( 8 < val_cnt) val[ 8][wi % N] = _val8;
			if ( 9 < val_cnt) val[ 9][wi % N] = _val9;
			if (10 < val_cnt) val[10][wi % N] = _val10;
			if (11 < val_cnt) val[11][wi % N] = _val11;
			if (12 < val_cnt) val[12][wi % N] = _val12;
			if (13 < val_cnt) val[13][wi % N] = _val13;
			if (14 < val_cnt) val[14][wi % N] = _val14;
			if (15 < val_cnt) val[15][wi % N] = _val15;
			if (16 < val_cnt) val[16][wi % N] = _val16;
			if (17 < val_cnt) val[17][wi % N] = _val17;
			if (18 < val_cnt) val[18][wi % N] = _val18;
			if (19 < val_cnt) val[19][wi % N] = _val19;
			if (20 < val_cnt) val[20][wi % N] = _val20;
			if (21 < val_cnt) val[21][wi % N] = _val21;
#ifdef __CUDA_ARCH__
//			__threadfence();
			memoryfence<fencetype>();
#endif
			safe_to_read[wi % N] = 1;
		}
	}

	// called by entire warp
	// returns 0xFFFFFFFF if read is not possible
	// returns per-thread read index if read is safe
	__host__ __device__ inline uint32_t getreadidx(control_t& control)
	{
#ifdef __CUDA_ARCH__
		return control.warp_read_idx(safe_to_read);
#else
		return control.host_read_idx(safe_to_read);
#endif
	}

	template<size_t i>
	__host__ __device__ inline val_t get(uint32_t idx)
	{
		ASSERT_COMPILE_TIME(i<val_cnt);
		return val[i][idx % N];
	}
};

















/****** CAS ******/

// control logic using CAS version
template<size_t N>
class cyclic_buffer_control_cas_t
{
	public:

	static const uint32_t size = N;
	volatile uint32_t write_idx;
	volatile uint32_t written_idx;
	volatile uint32_t read_idx;

	__host__ __device__ inline void reset()
	{
		write_idx = 0;
		written_idx = 0;
		read_idx = 0;
	}

	// assumes entire warp calls this function with identical warp_to_write_mask
	// returns per-thread idx to write to
	__device__ inline uint32_t warp_write_idx(uint32_t warp_to_write_mask)
	{
		// warp: determine count and offset
		uint32_t count  = __popc(warp_to_write_mask);
		// thread ZERO has offset 0
		uint32_t offset = count - __popc(warp_to_write_mask >> (threadIdx.x&31));
		uint32_t baseidx = 0;
		// thread ZERO increases write_idx with count and obtains old value in baseidx
		if ((threadIdx.x&31)==0)
		{
			baseidx = atomicAdd((uint32_t*)&write_idx,count);
		}
		// thread ZERO passes baseidx to entire warp
		baseidx = __shfl(baseidx,0);
		return baseidx + offset;
	}

	// assumes entire warp calls this function with their per-thread idx to write to
	__device__ inline void warp_write_finish(uint32_t idx, uint32_t mask)
	{
		uint32_t count = __popc(mask);
		// thread ZERO updates written_idx
		// idx was old value of write_idx
		// waits till written_idx has old value of write_idx
		// and then increases it by count
		if ((threadIdx.x&31)==0)
		{
			while (idx != atomicCAS((uint32_t*)&written_idx, idx, idx+count) )
				;
		}
	}

	// assumes entire warp calls this function with identical inputs
	// if read is safe return per-thread idx to read from, returns 0xFFFFFFFF otherwise
	__device__ inline uint32_t warp_read_idx()
	{
		while (true)
		{
			// obtain warp-wide unique race-free value of read_idx
			uint32_t baseidx = __shfl(read_idx,0);
			uint32_t baseidxdist = __shfl(written_idx,0);
			// if not safe to read then return false
			if (baseidxdist-baseidx < 32)
			{
				return 0xFFFFFFFF;
			}
			uint32_t thrdidx = (baseidx+(threadIdx.x&31));
			// thread ZERO tries to increase read_idx with 32, baseidx2==baseidx in case of success, != otherwise
			uint32_t baseidx2;
			if ((threadIdx.x&31)==0)
			{
				baseidx2 = atomicCAS((uint32_t*)&read_idx, baseidx, baseidx+32);
			}
			// thread ZERO passes baseidx2 to entire warp
			baseidx2 = __shfl(baseidx2, 0);
			// if read_idx was successfully increased return (true,baseidx+offset)
			if (baseidx2 == baseidx)
			{
				return thrdidx;
			}
			// increase failed due to race, try again
		}
	}


	/**** HOST FUNCTIONS ASSUME EXCLUSIVE ACCESS FROM 1 HOST THREAD ****/
	__host__ inline uint32_t host_write_idx()
	{
		uint32_t i = write_idx;
		++write_idx;
		return i;
	}
	__host__ inline void host_write_finished()
	{
		++written_idx;
	}
	__host__ inline uint32_t host_read_idx()
	{
		if (read_idx == write_idx)
		{
			return 0xFFFFFFFF;
		}
		uint32_t i = read_idx;
		++read_idx;
		return i;
	}
};



template<size_t N, typename val_t = uint32_t, size_t val_cnt = 1, typename control_type = cyclic_buffer_control_cas_t<N>, int fencetype = 2 >
class cyclic_buffer_cas_t
{
	public:

	typedef control_type control_t;

	static const uint32_t size = N;
	static const uint32_t val_size = val_cnt;

	volatile val_t val[val_cnt][N];
	
	// called by entire warp
	__host__ __device__ inline void reset(control_t& control)
	{
		control.reset();
	}

	// called by entire warp
	__host__ __device__ inline void write(control_t& control, bool dowrite, val_t _val0 =0, val_t _val1 =0)
	{
#ifdef __CUDA_ARCH__
		uint32_t mask = __ballot(dowrite);
		if (mask == 0) 
		{
			return;
		}
		uint32_t wi = control.warp_write_idx(mask);
		if (dowrite)
		{
			if ( 0 < val_cnt) val[ 0][wi % N] = _val0;
			if ( 1 < val_cnt) val[ 1][wi % N] = _val1;
//			__threadfence();
			memoryfence<fencetype>();
		}
		control.warp_write_finish(wi,mask);
#else
		if (dowrite)
		{
			uint32_t wi = control.host_write_idx();
			if (0 < val_cnt) val[0][wi % N] = _val0;
			if (1 < val_cnt) val[1][wi % N] = _val1;
			control.host_write_finished();
		}
#endif
	}


	// called by entire warp
	__host__ __device__ inline void write(control_t& control, bool dowrite
		, val_t _val0   , val_t _val1   , val_t _val2   , val_t _val3 =0, val_t _val4 =0, val_t _val5 =0, val_t _val6 =0, val_t _val7 =0, val_t _val8 =0, val_t _val9 =0
		, val_t _val10=0, val_t _val11=0, val_t _val12=0, val_t _val13=0, val_t _val14=0, val_t _val15=0, val_t _val16=0, val_t _val17=0, val_t _val18=0, val_t _val19=0
		, val_t _val20=0, val_t _val21=0 
		/*, val_t _val22=0, val_t _val23=0, val_t _val24=0, val_t _val25=0, val_t _val26=0, val_t _val27=0, val_t _val28=0, val_t _val29=0 */
		)
	{
#ifdef __CUDA_ARCH__
		uint32_t mask = __ballot(dowrite);
		if (mask == 0) 
		{
			return;
		}
		uint32_t wi = control.warp_write_idx(mask);
		if (dowrite)
		{
			if ( 0 < val_cnt) val[ 0][wi % N] = _val0;
			if ( 1 < val_cnt) val[ 1][wi % N] = _val1;
			if ( 2 < val_cnt) val[ 2][wi % N] = _val2;
			if ( 3 < val_cnt) val[ 3][wi % N] = _val3;
			if ( 4 < val_cnt) val[ 4][wi % N] = _val4;
			if ( 5 < val_cnt) val[ 5][wi % N] = _val5;
			if ( 6 < val_cnt) val[ 6][wi % N] = _val6;
			if ( 7 < val_cnt) val[ 7][wi % N] = _val7;
			if ( 8 < val_cnt) val[ 8][wi % N] = _val8;
			if ( 9 < val_cnt) val[ 9][wi % N] = _val9;
			if (10 < val_cnt) val[10][wi % N] = _val10;
			if (11 < val_cnt) val[11][wi % N] = _val11;
			if (12 < val_cnt) val[12][wi % N] = _val12;
			if (13 < val_cnt) val[13][wi % N] = _val13;
			if (14 < val_cnt) val[14][wi % N] = _val14;
			if (15 < val_cnt) val[15][wi % N] = _val15;
			if (16 < val_cnt) val[16][wi % N] = _val16;
			if (17 < val_cnt) val[17][wi % N] = _val17;
			if (18 < val_cnt) val[18][wi % N] = _val18;
			if (19 < val_cnt) val[19][wi % N] = _val19;
			if (20 < val_cnt) val[20][wi % N] = _val20;
			if (21 < val_cnt) val[21][wi % N] = _val21;
/*
			if (22 < val_cnt) val[22][wi % N] = _val22;
			if (23 < val_cnt) val[23][wi % N] = _val23;
			if (24 < val_cnt) val[24][wi % N] = _val24;
			if (25 < val_cnt) val[25][wi % N] = _val25;
			if (26 < val_cnt) val[26][wi % N] = _val26;
			if (27 < val_cnt) val[27][wi % N] = _val27;
			if (28 < val_cnt) val[28][wi % N] = _val28;
			if (29 < val_cnt) val[29][wi % N] = _val29;
*/
//			__threadfence();
			memoryfence<fencetype>();
		}
		control.warp_write_finish(wi,mask);
#else
		if (dowrite)
		{
			uint32_t wi = control.host_write_idx();
			if ( 0 < val_cnt) val[ 0][wi % N] = _val0;
			if ( 1 < val_cnt) val[ 1][wi % N] = _val1;
			if ( 2 < val_cnt) val[ 2][wi % N] = _val2;
			if ( 3 < val_cnt) val[ 3][wi % N] = _val3;
			if ( 4 < val_cnt) val[ 4][wi % N] = _val4;
			if ( 5 < val_cnt) val[ 5][wi % N] = _val5;
			if ( 6 < val_cnt) val[ 6][wi % N] = _val6;
			if ( 7 < val_cnt) val[ 7][wi % N] = _val7;
			if ( 8 < val_cnt) val[ 8][wi % N] = _val8;
			if ( 9 < val_cnt) val[ 9][wi % N] = _val9;
			if (10 < val_cnt) val[10][wi % N] = _val10;
			if (11 < val_cnt) val[11][wi % N] = _val11;
			if (12 < val_cnt) val[12][wi % N] = _val12;
			if (13 < val_cnt) val[13][wi % N] = _val13;
			if (14 < val_cnt) val[14][wi % N] = _val14;
			if (15 < val_cnt) val[15][wi % N] = _val15;
			if (16 < val_cnt) val[16][wi % N] = _val16;
			if (17 < val_cnt) val[17][wi % N] = _val17;
			if (18 < val_cnt) val[18][wi % N] = _val18;
			if (19 < val_cnt) val[19][wi % N] = _val19;
			if (20 < val_cnt) val[20][wi % N] = _val20;
			if (21 < val_cnt) val[21][wi % N] = _val21;
/*
			if (22 < val_cnt) val[22][wi % N] = _val22;
			if (23 < val_cnt) val[23][wi % N] = _val23;
			if (24 < val_cnt) val[24][wi % N] = _val24;
			if (25 < val_cnt) val[25][wi % N] = _val25;
			if (26 < val_cnt) val[26][wi % N] = _val26;
			if (27 < val_cnt) val[27][wi % N] = _val27;
			if (28 < val_cnt) val[28][wi % N] = _val28;
			if (29 < val_cnt) val[29][wi % N] = _val29;
*/
			control.host_write_finished();
		}
#endif
	}

	// called by entire warp
	// returns 0xFFFFFFFF if read is not possible
	// returns per-thread read index if read is safe
	__host__ __device__ inline uint32_t getreadidx(control_t& control)
	{
#ifdef __CUDA_ARCH__
		return control.warp_read_idx();
#else
		return control.host_read_idx();
#endif
	}

	template<size_t i>
	__host__ __device__ inline val_t get(uint32_t idx)
	{
		ASSERT_COMPILE_TIME(i<val_cnt);
		return val[i][idx % N];
	}
};






/***** TEMPORARY BUFFER ******/
class warp_tmp_buf_t
{
	public:
	// temporary buffer for 64 2-word elems: two halves of 32 elems
	// whenever one half is full it is flushed to the main buffer
	uint32_t val1[64];
	uint32_t val2[64];
	volatile char idx;

	__device__ void reset()
	{
		idx = 0;
	}

	// store 1-word elements in temporary shared buffer, flush to global buffer for each 32 values
	template<typename buffer_t, typename control_t>
	__device__ inline void write1(bool dowrite, uint32_t _val1, buffer_t& buf, control_t& ctrl)
	{
		uint32_t mask = __ballot(dowrite);
		if (mask == 0)
		{
			return;
		}

		uint32_t count = __popc(mask);
		uint32_t offset = count - __popc(mask >> (threadIdx.x&31));
		uint32_t baseidx = idx;
		if ((threadIdx.x&31)==0)
		{
			idx = (idx + (char)(count)) % 64;
		}

		if (dowrite)
		{
			val1[(baseidx + offset) % 64] = _val1;
		}

		// flush 'full' half if we cross over halves
		if ((idx^baseidx)&32)
		{
			baseidx &= 32; // point to start of 'full' half
			buf.write(ctrl, true, val1[baseidx + (threadIdx.x&31)] );
		}
	}
	
	template<class buffer_t, class control_t>
	__device__ inline void write2(bool dowrite, uint32_t _val1, uint32_t _val2, buffer_t& buf, control_t& ctrl)
	{
		uint32_t mask = __ballot(dowrite);
		if (mask == 0)
		{
			return;
		}

		uint32_t count = __popc(mask);
		uint32_t offset = count - __popc(mask >> (threadIdx.x&31));
		uint32_t baseidx = idx;
		if ((threadIdx.x&31)==0)
		{
			idx = (idx + (char)(count)) % 64;
		}

		if (dowrite)
		{
			val1[(baseidx + offset) % 64] = _val1;
			val2[(baseidx + offset) % 64] = _val2;
		}

		// flush 'full' half if we cross over halves
		if ((idx^baseidx)&32)
		{
			baseidx &= 32; // point to start of 'full' half
			buf.write(ctrl, true, val1[baseidx + (threadIdx.x&31)], val2[baseidx + (threadIdx.x&31)] );
		}
	}

	template<typename buffer_t, typename control_t>
	__device__ inline void flush1(buffer_t& buf, control_t& ctrl)
	{
		uint32_t baseidx = (uint32_t)(idx) & 32;
		if (idx != baseidx)
		{
			buf.write(ctrl, (threadIdx.x&31)<(idx-baseidx), val1[baseidx + (threadIdx.x&31)] );
		}
		reset();
	}		

	template<typename buffer_t, typename control_t>
	__device__ inline void flush2(buffer_t& buf, control_t& ctrl)
	{
		uint32_t baseidx = (uint32_t)(idx) & 32;
		if (idx != baseidx)
		{
			buf.write(ctrl, (threadIdx.x&31)<(idx-baseidx), val1[baseidx + (threadIdx.x&31)], val2[baseidx + (threadIdx.x&31)] );
		}
		reset();
	}		


};
