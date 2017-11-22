/*****
  Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
            (C) 2016 Pierre Karpman
            
  This file is part of sha1freestart80 source-code and released under the MIT License
*****/

#include <fstream>
#include <iomanip>
#include <algorithm>

#include "neutral_bits_packing.hpp"
#include "main.hpp"

using namespace hc;

int cuda_device, cuda_blocks, cuda_threads_per_block;
vector<string> inputfile;
string outputfile;
bool disable_backwards_filter = false;
int cuda_scheduler;
int max_basesols;
 
namespace maincpp {
#include "tables.hpp"
}
using namespace maincpp;


bool verifyQ60(const q60sol_t& q60sol)
{
#if 1
	for (int t = -4; t < 20; ++t)
	{
		cout << "Q" << t << "\t: ";
		uint32_t Q1 = Qset1mask[Qoffset+t];
		uint32_t Q2 = Q1 + dQ[Qoffset+t];
		uint32_t XQ = Q1^Q2;
		for (int b = 31; b >= 0; --b)
		{
			if (Qcondmask[Qoffset+t]&(1<<b))
			{
				bool flip   = (Qset1mask[Qoffset+t]&(1<<b))!=0;
				bool prev   = (Qprevmask[Qoffset+t]&(1<<b))!=0;
				bool prevr  = (Qprevrmask[Qoffset+t]&(1<<b))!=0;
				bool prev2r = (Qprev2rmask[Qoffset+t]&(1<<b))!=0;
				bool next   = (Qnextmask[Qoffset+t]&(1<<b))!=0;
				unsigned cnt = 0;
				if (prev) ++cnt;
				if (prevr) ++cnt;
				if (prev2r) ++cnt;
				if (next) ++cnt;
				if (cnt>1)
				{
					cout << "cnt>1: " << cnt << " " << prev << prevr << prev2r << next << endl;
				}
				if (cnt == 0)
				{
					if (XQ & (1<<b))
						cout << (flip?"-":"+");
					else
						cout << (flip?"1":"0");
				}
				if (prev) cout << (!flip?"^":"!");
				if (prevr) cout << (!flip?"r":"R");
				if (prev2r) cout << (!flip?"s":"S");
				if (next) cout << (!flip?"v":"Y");
			}
			else
			{
				cout << ".";
			}
		}
		cout << endl;
	}
#endif

	uint32_t m1[80];
	uint32_t m2[80];
	uint32_t Q1[86];
	uint32_t Q2[86];
	for (int t = 56; t <= 60; ++t)
	{
		Q2[Qoffset + t] = Q1[Qoffset + t] = q60sol.Q[t - 56];
	}
	for (int t = 44; t < 44 + 16; ++t)
	{
		m2[t] = q60sol.m[t - 44];
	}
	sha1_me_generalised(m2, 44);

	for (int t = 0; t < 80; ++t)
	{
		m1[t] = m2[t] ^ DV_DW[t];
#if 1
		if (t < 21)
		{
			cout << "Delta m" << t << ": " << hex << setw(8) << setfill('0') << m2[t] - m1[t] << dec << endl;
		}
#endif
	}

	for (int t = 60; t < 80; ++t)
	{
		sha1_step(t, Q1, m1);
		sha1_step(t, Q2, m2);
	}
	for (int t = 59; t >= 0; --t)
	{
		sha1_step_bw(t, Q1, m1);
		sha1_step_bw(t, Q2, m2);
	}
	cout << "=======================" << endl;


	for (int t = -4; t <= 80; ++t)
	{
		cout << "dQ" << t << "\t: " << hex << setw(8) << setfill('0') << (Q2[Qoffset + t] - Q1[Qoffset + t]) << dec << endl;
	}


	bool okay = true;
	for (int t = -4; t <= 80; ++t)
	{
		uint32_t Qval = Qset1mask[Qoffset+t];
		if (t-1 >= -4)
		{
			Qval ^= (Qprevmask[Qoffset+t] & Q1[Qoffset+t-1]);
			Qval ^= (Qprevrmask[Qoffset+t] & rotate_left(Q1[Qoffset+t-1],30));
		}
		if (t-2 >= -4)
		{
			Qval ^= (Qprev2rmask[Qoffset+t] & rotate_left(Q1[Qoffset+t-2],30));
		}
		Qval ^= (Qnextmask[Qoffset+t] & Q1[Qoffset+t+1]);
		if ( (Qval ^ Q1[Qoffset+t]) & Qcondmask[Qoffset+t] )
		{
			cout << "Q" << t << "\t: cond bad: " << hex << setw(8) << setfill('0') << ((Qval^Q1[Qoffset+t])&Qcondmask[Qoffset+t]) << dec << endl;
			okay = false;
		}
	}
	// verify message bitrelations
#if 0
	for (unsigned r = 0; r < msgbitrels16_size; ++r)
	{
		uint32_t w = msgbitrels16[r][16];
		for (unsigned t = mainblockoffset; t < mainblockoffset+16; ++t)
		{
			w ^= m1[t] & msgbitrels16[r][t-mainblockoffset];
		}
		if (0 != (hw(w)&1) )
		{
			std::cout << "16 bitrelation " << r << " is not satisfied!" << std::endl;
			okay = false;
			for (unsigned t = mainblockoffset; t < mainblockoffset+16; ++t)
			{
				uint32_t relt = msgbitrels16[r][t-mainblockoffset];
				for (unsigned b = 0; b < 32; ++b)
					if ((relt>>b)&1)
						cout << "W" <<t << "[" << b << "]";
			}
			cout << " = " << (msgbitrels16[r][16]&1) << endl;
		}
	}
#endif
	// verify message bitrelations again
	for (unsigned r = 0; r < msgbitrels80_size; ++r)
	{
		uint32_t w = msgbitrels80[r][80];
		for (unsigned t = 0; t < 80; ++t)
		{
			w ^= m1[t] & msgbitrels80[r][t];
		}
		if (0 != (hw(w)&1) )
		{
			std::cout << "80 bitrelation " << r << " is not satisfied!" << std::endl;
			okay = false;
			for (unsigned t = 0; t < 80; ++t)
			{
				uint32_t relt = msgbitrels80[r][t];
				for (unsigned b = 0; b < 32; ++b)
					if ((relt>>b)&1)
						cout << "^ W" <<t << "[" << b << "]";
			}
			cout << " = " << (msgbitrels80[r][80]&1) << endl;
		}
	}
	for (int t = -4; t <= 30; ++t)
	{
		if (dQ[Qoffset + t] != Q2[Qoffset + t] - Q1[Qoffset + t])
		{
			cout << "dQ" << t << " is bad!: " << hex << dQ[Qoffset + t] << "!=" << hex << (Q2[Qoffset + t] - Q1[Qoffset + t]) << dec << endl;
			okay = false;
		}
	}
	for (int t = 56; t <= 60; ++t)
	{
		if (0 != Q2[Qoffset + t] - Q1[Qoffset + t])
		{
			cout << "dQ" << t << " is bad!: " << hex << 0 << "!=" << hex << (Q2[Qoffset + t] - Q1[Qoffset + t]) << dec << endl;
			okay = false;
		}
	}
	cout << endl;
	uint32_t ihv1[5] = { Q1[4], Q1[3], rotate_left(Q1[2],30), rotate_left(Q1[1],30), rotate_left(Q1[0],30) };
	uint32_t ihv2[5] = { Q2[4], Q2[3], rotate_left(Q2[2],30), rotate_left(Q2[1],30), rotate_left(Q2[0],30) };
	ihv1[0] += Q1[Qoffset+80];
	ihv1[1] += Q1[Qoffset+79];
	ihv1[2] += rotate_left(Q1[Qoffset+78],30);
	ihv1[3] += rotate_left(Q1[Qoffset+77],30);
	ihv1[4] += rotate_left(Q1[Qoffset+76],30);
	ihv2[0] += Q2[Qoffset+80];
	ihv2[1] += Q2[Qoffset+79];
	ihv2[2] += rotate_left(Q2[Qoffset+78],30);
	ihv2[3] += rotate_left(Q2[Qoffset+77],30);
	ihv2[4] += rotate_left(Q2[Qoffset+76],30);
	if (ihv1[0] == ihv2[0]) cout << "ihv[0] ok !" << endl;
	if (ihv1[1] == ihv2[1]) cout << "ihv[1] ok !" << endl;
	if (ihv1[2] == ihv2[2]) cout << "ihv[2] ok !" << endl;
	if (ihv1[3] == ihv2[3]) cout << "ihv[3] ok !" << endl;
	if (ihv1[4] == ihv2[4]) cout << "ihv[4] ok !" << endl;

	if (ihv1[0]==ihv2[0] && ihv1[1]==ihv2[1] && ihv1[2]==ihv2[2] && ihv1[3]==ihv2[3] && ihv1[4]==ihv2[4])
	{
		uint32_t ihv1a[5] = { Q1[4], Q1[3], rotate_left(Q1[2],30), rotate_left(Q1[1],30), rotate_left(Q1[0],30) };
		uint32_t ihv2a[5] = { Q2[4], Q2[3], rotate_left(Q2[2],30), rotate_left(Q2[1],30), rotate_left(Q2[0],30) };
		cout << "Found solution!!" << endl;
		for (int t = 0; t < 5; ++t)
		{
			cout << "IV " <<t << "\t: 0x" << hex << setw(8) << setfill('0') << ihv1a[t] << dec << endl;
			cout << "IV'" <<t << "\t: 0x" << hex << setw(8) << setfill('0') << ihv2a[t] << dec << endl;
		}
		for (int t = 0; t < 16; ++t)
		{
			cout << "m " <<t << "\t: 0x" << hex << setw(8) << setfill('0') << m1[t] << dec << endl;
			cout << "m'" <<t << "\t: 0x" << hex << setw(8) << setfill('0') << m2[t] << dec << endl;
		}
		for (int t = 0; t < 5; ++t)
		{
			cout << "CV " <<t << "\t: 0x" << hex << setw(8) << setfill('0') << ihv1[t] << dec << endl;
			cout << "CV'" <<t << "\t: 0x" << hex << setw(8) << setfill('0') << ihv2[t] << dec << endl;
		}
	}

	return okay;
}


void verifyQ60(const string& filename)
{
	ifstream ifs(filename.c_str());
	if (!ifs)
	{
		cout << "Error: failed to open '" << filename << "'!" << endl;
		return;
	}
	vector<q60sol_t> q60sols;
	while (!!ifs)
	{
		std::string line;
		getline(ifs, line);
		if (line.substr(0,2) == "Q!")
		{
			q60sols.push_back(decode_q60sol(line));
		}
	}

	cout << "Loaded " << q60sols.size() << " Q60 solutions from '" << filename << "'." << endl;
	ifs.close();

	size_t okcnt = 0;
	for (size_t i = 0; i < q60sols.size(); ++i)
	{
		if (verifyQ60(q60sols[i]))
		{
			++okcnt;
		}
	}
	cout << "Verified: " << okcnt << " OK out of " << q60sols.size() << "." << endl;
}



///////////////////////////////////////////////////////////
//struct basesol_t {
//        uint32_t Q[6];  // Q12,..,Q17
//        uint32_t m[16]; // W5,...,W20
//};
bool operator==(const basesol_t& l, const basesol_t& r)
{
	for (int i = 0; i < 6; ++i)
		if (l.Q[i] != r.Q[i])
			return false;
	for (int i = 0; i < 16; ++i)
		if (l.m[i] != r.m[i])
			return false;
	return true;
}
bool operator<(const basesol_t& l, const basesol_t& r)
{
	for (int i = 0; i < 6; ++i)
		if (l.Q[i] != r.Q[i])
			return l.Q[i] < r.Q[i];
	for (int i = 0; i < 16; ++i)
		if (l.m[i] != r.m[i])
			return l.m[i] < r.m[i];
	return false;
}
vector<basesol_t> basesols;

void save_basesols(const string& filename)
{
	if (filename.empty())
	{
		return;
	}
	ofstream ofs(filename.c_str());
	if (!ofs)
	{
		cerr << "Cannot open file '" << filename << "'!" << endl;
		return;
	}
	for (unsigned i = 0; i + 31 < basesols.size(); i += 32)
	{
		for (unsigned j = 0; j < 32; ++j)
			ofs << encode_basesol(basesols[i + j]) << " ";
		ofs << endl;
	}
	cout << "Written " << (basesols.size()&~size_t(31)) << " basesols to '" << filename << "'." << endl;
}

vector<basesol_t> load_basesols(const string& filename)
{
	std::vector<basesol_t> ret;
	ifstream ifs(filename.c_str());
	if (!ifs)
		return ret;
	while (!!ifs)
	{
		std::string line;
		getline(ifs, line);

		size_t pos = line.find("B!");
		while (pos < line.size())
		{
			size_t pos2 = line.find(" ", pos);
			ret.push_back(decode_basesol(line.substr(pos, pos2 - pos)));
			pos = line.find("B!", pos2);
		}
	}

	cout << "Loaded " << ret.size() << " basesols from '" << filename << "'." << endl;

	for (size_t i = 0; i < ret.size();)
	{
		if (verify(ret[i]))
		{
			++i;
		}
		else
		{
			swap(ret[i], ret.back());
			ret.pop_back();
		}
	}

	cout << "Filter " << ret.size() << " basesols from '" << filename << "'." << endl;
	return ret;
}

void load_basesols(const vector<string>& inputfile)
{
	for (vector<string>::const_iterator it = inputfile.begin(); it != inputfile.end(); ++it)
	{
		vector<basesol_t> tmp;
		tmp = load_basesols(*it);
		basesols.insert(basesols.end(), tmp.begin(), tmp.end());
	}
	if (!basesols.empty())
	{
		sort(basesols.begin(), basesols.end());
		basesols.erase( unique(basesols.begin(),basesols.end()), basesols.end());
		for (size_t i = 0; i < basesols.size(); ++i)
		{
			size_t j = xrng128() % basesols.size();
			std::swap(basesols[i], basesols[j]);
		}
		cout << "Loaded " << basesols.size() << " basesols from inputfile(s)." << endl;
	}
}

/*extern*/ bool BASESOL_OK = true;
void found_basesol(uint32_t main_m1[80], uint32_t main_Q1[85], unsigned mainblockoffset)
{
	basesol_t basesol;

	for (int t = 12; t <= 17; ++t)
		basesol.Q[t-12] = main_Q1[4+t];

	// compute ALL W0,...,W79
	sha1_me_generalised(main_m1, mainblockoffset);

	for (int t = 5; t < 5+16; ++t)
		basesol.m[t-5] = main_m1[t];

	BASESOL_OK = verify(basesol);
	if (BASESOL_OK)
	{
		basesols.push_back(basesol);
		if (hc::hw(basesols.size()) == 1)
		{
			cout << "{" << basesols.size() << "}" << flush;
		}
		if (max_basesols > 0 && basesols.size() == max_basesols)
		{
			if (!outputfile.empty())
				save_basesols(outputfile);
			exit(0);
		}
		// save every 32 basesols
		if ((basesols.size()&31) == 0 && !outputfile.empty())
		{
			save_basesols(outputfile);
		}
	}
}


void sanity_checks()
{
	if (W14NBALLM != (W14NBQ18M | W14NBQ19M)) cout << "W14NBALLM bad" << endl;
	if (W14NBALLM != (W14NBQ18M ^ W14NBQ19M)) cout << "W14NBQ*M bad" << endl;

	if (W15NBALLM != (W15NBQ18M | W15NBQ19M | W15NBQ20M)) cout << "W15NBALLM bad" << endl;
	if (W15NBALLM != (W15NBQ18M ^ W15NBQ19M ^ W15NBQ20M)) cout << "W15NQ*M bad" << endl;

	if (W16NBALLM != (W16NBQ19M | W16NBQ20M)) cout << "W16NBALLM bad" << endl;
	if (W16NBALLM != (W16NBQ19M ^ W16NBQ20M)) cout << "W16NQ*M bad" << endl;

	if (W17NBALLM != (W17NBQ20M | W17NBQ21M)) cout << "W17NBALLM bad" << endl;
	if (W17NBALLM != (W17NBQ20M ^ W17NBQ21M)) cout << "W17NQ*M bad" << endl;

	if (W17NBALLM != (W17NBBASM | W17NBEXTM)) cout << "W17NBALLM bad" << endl;
	if (W17NBALLM != (W17NBBASM ^ W17NBEXTM)) cout << "W17NBASEXTM bad" << endl;

	if (W18NBALLM != (W18NBQ21M | W18NBQ22M | W18NBQ23M)) cout << "W18NBALLM bad" << endl;
	if (W18NBALLM != (W18NBQ21M ^ W18NBQ22M ^ W18NBQ23M)) cout << "W18NQ*M bad" << endl;

	if (W19NBALLM != (W19NBQ22M | W19NBQ23M | W19NBQ24M)) cout << "W19NBALLM bad" << endl;
	if (W19NBALLM != (W19NBQ22M ^ W19NBQ23M ^ W19NBQ24M)) cout << "W19NQ*M bad" << endl;

	if (W20NBALLM != (W20NBQ23M | W20NBQ24M | W20NBQ25M)) cout << "W20NBALLM bad" << endl;
	if (W20NBALLM != (W20NBQ23M ^ W20NBQ24M ^ W20NBQ25M)) cout << "W20NQ*M bad" << endl;

}


int main(int argc, char** argv)
{
	sanity_checks();

	// extern int cuda_device, cuda_blocks, cuda_threads_per_block;
	// extern vector<string> inputfile;
	// extern string outputfile;
	// extern bool disable_backwards_filter;
	// extern int cuda_scheduler;

	// DEFAULTS
	cuda_device = 0; // default is to use first CUDA device
	cuda_blocks = -1; // automatic
	cuda_threads_per_block = -1; // automatic
	cuda_scheduler = 3; // 0=auto, 1=spin, 2=yield, 3=blockingsync

	std::string seedstr;
	const char* seedchars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	for (int i = 0; i < 22; ++i)
		seedstr += seedchars[xrng128() % 62];


	// Parse arguments
	
	enum { Help, Query, Benchmark, Attack, Basesol, Verify } command = Help;
	for (int argi = 1; argi < argc; ++argi)
	{
		std::string stri(argv[argi]);
		if (stri == "-h" || stri == "--help")
		{
		}
		else if (stri == "-g" || stri == "--genbasesol")
		{
			command = Basesol;
		}
		else if (stri == "-v" || stri == "--verify")
		{
			command = Verify;
		}
		else if (stri == "-i" || stri == "--input")
		{
			if (++argi < argc)
				inputfile.push_back(std::string(argv[argi]));
		}
		else if (stri == "-o" || stri == "--output")
		{
			if (++argi < argc)
				outputfile = std::string(argv[argi]);
		}
		else if (stri == "-r" || stri == "--rndseed")
		{
			if (++argi < argc)
				seedstr = std::string(argv[argi]);
		}
		else if (stri == "-m" || stri == "--maxbasesol")
		{
			if (++argi < argc)
				max_basesols = atoi(argv[argi]);
		}
#ifndef NOCUDA
		else if (stri == "-q" || stri == "--query")
		{
			command = Query;
		}
		else if (stri == "-b" || stri == "--benchmark")
		{
			command = Benchmark;
		}
		else if (stri == "-a" || stri == "--attack")
		{
			command = Attack;
		}
		else if (stri == "-d" || stri == "--device")
		{
			if (++argi < argc)
				cuda_device = atoi(argv[argi]);
		}
		else if (stri == "-s" || stri == "--cudasched")
		{
			if (++argi < argc)
				cuda_scheduler = atoi(argv[argi]);
		}
		else if (stri == "--gpublocks")
		{
			if (++argi < argc)
				cuda_blocks = atoi(argv[argi]);
		}
		else if (stri == "--gputhreads")
		{
			if (++argi < argc)
				cuda_threads_per_block = atoi(argv[argi]);
		}
#endif
	}

	// Process seed str
	cout << "Current seed string: " << seedstr << endl;
	seed(0);
	for (int i = 0; i < seedstr.size(); ++i)
	{
		uint32_t x = uint32_t(seedstr[i]);
		x = rotate_right( (x<<24)|(x<<16)|(x<<8)|x, (i&31));
		addseed( x ^ i );
	}

	timer runtime(true);
	
	try {
	switch (command)
	{
		case Help:
			cout << "Usage:" << endl
				<< "[-h|--help]              Show help" << endl
				<< "[-g|--genbasesol]        Generate basesolutions" << endl
				<< "[-v|--verify]            Verify 60-step solutions" << endl
				<< "[-i|--input <file>]      Add inputfile" << endl
				<< "[-o|--output <file>]     Set outputfile" << endl
				<< "[-r|--rndseed <seed>]    Set seed for PRNG" << endl
				<< "[-m|--maxbasesols <nr>]  Set max basesols to generate" << endl
#ifndef NOCUDA
				<< "(CUDA options)" << endl
				<< "[-q|--query]             Query CUDA devices" << endl
				<< "[-b|--benchmark]         Benchmark CUDA device" << endl
				<< "[-a|--attack]            Start CUDA attack" << endl
				<< "[-d|--device <nr>]       Sets CUDA device to use" << endl
				<< "[-s|--cudasched <nr>]    Sets CUDA wait method: 0=auto, 1=spin, 2=yield, 3=blockingsync(default)" << endl
				<< "[--gpublocks <nr>]       Sets number of blocks to use, -1=automatic(default)" << endl
				<< "[--gputhreads <nr>]      Sets number of threads per block, -1=automatic(default)" << endl
#endif
				<< endl;
			break;
#ifndef NOCUDA
		case Query:
			cuda_query();
			break;
		case Benchmark:
			gpusha1benchmark();
			break;
		case Attack:
			if (!inputfile.empty())
			{
				load_basesols(inputfile);
				cuda_main(basesols);
			}
			else
			{
				cout << "Error: no inputfile with basesolutions given!" << endl; 
			}
			break;
#endif
		case Basesol:
			if (!inputfile.empty())
			{
				load_basesols(inputfile);
				if (!basesols.empty())
				{
					save_basesols(outputfile);
				}
			}
			else
			{
				if (outputfile.empty())
				{
					cout << "Warning: no outputfile given, results will be lost." << endl;
				}
				start_attack(); // never returns, uses exit(0)
			}
			break;
		case Verify:
			if (!inputfile.empty())
			{
				for (size_t i = 0; i < inputfile.size(); ++i)
				{
					verifyQ60(inputfile[i]);
				}
			}
			else
			{
				cout << "Error: no inputfile with Q60 solutions given!" << endl;
			}
			break;
		
	} // switch(command)
	} // try
	catch (std::exception& e)
	{
		cerr << "Caught exception:" << endl << e.what() << endl;
	}
	cout << "Runtime: " << runtime.time() << endl;
	return 0;
}
