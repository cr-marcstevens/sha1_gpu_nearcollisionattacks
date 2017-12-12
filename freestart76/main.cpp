/*****
  Copyright (C) 2015 Pierre Karpman, INRIA France/Nanyang Technological University Singapore (-2016), CWI (2016/2017), L'Universite Grenoble Alpes (2017-)
            (C) 2015 Thomas Peyrin, Nanyang Technological University Singapore
            (C) 2015 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1_gpu_nearcollisionattacks source-code and released under the MIT License
*****/

#include "main.hpp"

namespace maincpp {
#include "tables.hpp"
}
using namespace maincpp;

#include "sha1detail.hpp"
#include "rng.hpp"

#include <program_options.hpp>
#include <timer.hpp>

#include <fstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <mutex>
#include <cstdio>
#include <algorithm>

using namespace hashclash;
using namespace std;
namespace po = program_options;

int cuda_device, cuda_blocks, cuda_threads_per_block;
vector<string> inputfile;
string outputfile;
bool disable_backwards_filter = false;
int cuda_scheduler;
int max_basesols;
 



bool verifyQ56(const q56sol_t& q56sol)
{
#if 0
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
	uint32_t Q1[85];
	uint32_t Q2[85];
	for (int t = 32; t <= 36; ++t)
	{
		Q2[Qoffset + t] = Q1[Qoffset + t] = q56sol.Q[t - 32];
	}
	for (int t = 20; t < 20 + 16; ++t)
	{
		m1[t] = q56sol.m[t - 20];
	}
	sha1_me_generalised(m1, 20);

	for (int t = 0; t < 80; ++t)
	{
		m2[t] = m1[t] ^ DV_DW[t];
#if 0
		if (t < 21)
		{
			cout << "Delta m" << t << ": " << hex << setw(8) << setfill('0') << m2[t] - m1[t] << dec << endl;
		}
#endif
	}

	for (int t = 36; t < 76; ++t)
	{
		sha1_step(t, Q1, m1);
		sha1_step(t, Q2, m2);
	}
	for (int t = 35; t >= 0; --t)
	{
		sha1_step_bw(t, Q1, m1);
		sha1_step_bw(t, Q2, m2);
	}
	cout << "=======================" << endl;


	for (int t = -4; t <= 76; ++t)
	{
		cout << "dQ" << t << "\t: " << hex << setw(8) << setfill('0') << (Q2[Qoffset + t] - Q1[Qoffset + t]) << dec << endl;
	}


	bool okay = true;
	for (int t = -4; t <= 76; ++t)
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
//			cout << "Q" << t << "\t: cond bad: " << hex << setw(8) << setfill('0') << ((Qval^Q1[Qoffset+t])&Qcondmask[Qoffset+t]) << dec << endl;
			okay = false;
		}
	}
	// verify message bitrelations
	for (unsigned r = 0; r < msgbitrels16_size; ++r)
	{
		uint32_t w = msgbitrels16[r][16];
		for (unsigned t = mainblockoffset; t < mainblockoffset+16; ++t)
		{
			w ^= m1[t] & msgbitrels16[r][t-mainblockoffset];
		}
		if (okay && 0 != (hw(w)&1) )
		{
//			std::cout << "16 bitrelation " << r << " is not satisfied!" << std::endl;
			okay = false;
		}
	}
	// verify message bitrelations again
	for (unsigned r = 0; r < msgbitrels80_size; ++r)
	{
		uint32_t w = msgbitrels80[r][80];
		for (unsigned t = 0; t < 80; ++t)
		{
			w ^= m1[t] & msgbitrels80[r][t];
		}
		if (okay && 0 != (hw(w)&1) )
		{
//			std::cout << "80 bitrelation " << r << " is not satisfied!" << std::endl;
			okay = false;
		}
	}
	for (int t = -4; t <= 36; ++t)
	{
		if (dQ[Qoffset + t] != Q2[Qoffset + t] - Q1[Qoffset + t])
		{
//			cout << "dQ" << t << " is bad!" << endl;
			okay = false;
		}
	}
	for (int t = 52; t <= 56; ++t)
	{
		if (0 != Q2[Qoffset + t] - Q1[Qoffset + t])
		{
//			cout << "dQ" << t << " is bad!" << endl;
			okay = false;
		}
	}
	cout << endl;
	uint32_t ihv1[5] = { Q1[4], Q1[3], rotate_left(Q1[2],30), rotate_left(Q1[1],30), rotate_left(Q1[0],30) };
	uint32_t ihv2[5] = { Q2[4], Q2[3], rotate_left(Q2[2],30), rotate_left(Q2[1],30), rotate_left(Q2[0],30) };
	ihv1[0] += Q1[Qoffset+76];
	ihv1[1] += Q1[Qoffset+75];
	ihv1[2] += rotate_left(Q1[Qoffset+74],30);
	ihv1[3] += rotate_left(Q1[Qoffset+73],30);
	ihv1[4] += rotate_left(Q1[Qoffset+72],30);
	ihv2[0] += Q2[Qoffset+76];
	ihv2[1] += Q2[Qoffset+75];
	ihv2[2] += rotate_left(Q2[Qoffset+74],30);
	ihv2[3] += rotate_left(Q2[Qoffset+73],30);
	ihv2[4] += rotate_left(Q2[Qoffset+72],30);
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

bool verifyQ56_new(const q56sol_t& q56sol)
{
#if 0
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
	uint32_t Q1[85];
	uint32_t Q2[85];
	for (int t = 52; t <= 56; ++t)
	{
		Q2[Qoffset + t] = Q1[Qoffset + t] = q56sol.Q[t - 52];
	}
	for (int t = 40; t < 40 + 16; ++t)
	{
		m2[t] = q56sol.m[t - 40];
	}
	sha1_me_generalised(m2, 40);

	for (int t = 0; t < 80; ++t)
	{
		m1[t] = m2[t] ^ DV_DW[t];
#if 0
		if (t < 21)
		{
			cout << "Delta m" << t << ": " << hex << setw(8) << setfill('0') << m2[t] - m1[t] << dec << endl;
		}
#endif
	}

	for (int t = 56; t < 76; ++t)
	{
		sha1_step(t, Q1, m1);
		sha1_step(t, Q2, m2);
	}
	for (int t = 55; t >= 0; --t)
	{
		sha1_step_bw(t, Q1, m1);
		sha1_step_bw(t, Q2, m2);
	}
	cout << "=======================" << endl;


	for (int t = -4; t <= 76; ++t)
	{
		cout << "dQ" << t << "\t: " << hex << setw(8) << setfill('0') << (Q2[Qoffset + t] - Q1[Qoffset + t]) << dec << endl;
	}


	bool okay = true;
	for (int t = -4; t <= 76; ++t)
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
//			cout << "Q" << t << "\t: cond bad: " << hex << setw(8) << setfill('0') << ((Qval^Q1[Qoffset+t])&Qcondmask[Qoffset+t]) << dec << endl;
			okay = false;
		}
	}
	// verify message bitrelations
	for (unsigned r = 0; r < msgbitrels16_size; ++r)
	{
		uint32_t w = msgbitrels16[r][16];
		for (unsigned t = mainblockoffset; t < mainblockoffset+16; ++t)
		{
			w ^= m1[t] & msgbitrels16[r][t-mainblockoffset];
		}
		if (okay && 0 != (hw(w)&1) )
		{
//			std::cout << "16 bitrelation " << r << " is not satisfied!" << std::endl;
			okay = false;
		}
	}
	// verify message bitrelations again
	for (unsigned r = 0; r < msgbitrels80_size; ++r)
	{
		uint32_t w = msgbitrels80[r][80];
		for (unsigned t = 0; t < 80; ++t)
		{
			w ^= m1[t] & msgbitrels80[r][t];
		}
		if (okay && 0 != (hw(w)&1) )
		{
//			std::cout << "80 bitrelation " << r << " is not satisfied!" << std::endl;
			okay = false;
		}
	}
	for (int t = -4; t <= 36; ++t)
	{
		if (dQ[Qoffset + t] != Q2[Qoffset + t] - Q1[Qoffset + t])
		{
//			cout << "dQ" << t << " is bad!" << endl;
			okay = false;
		}
	}
	for (int t = 52; t <= 56; ++t)
	{
		if (0 != Q2[Qoffset + t] - Q1[Qoffset + t])
		{
//			cout << "dQ" << t << " is bad!" << endl;
			okay = false;
		}
	}
	cout << endl;
	uint32_t ihv1[5] = { Q1[4], Q1[3], rotate_left(Q1[2],30), rotate_left(Q1[1],30), rotate_left(Q1[0],30) };
	uint32_t ihv2[5] = { Q2[4], Q2[3], rotate_left(Q2[2],30), rotate_left(Q2[1],30), rotate_left(Q2[0],30) };
	ihv1[0] += Q1[Qoffset+76];
	ihv1[1] += Q1[Qoffset+75];
	ihv1[2] += rotate_left(Q1[Qoffset+74],30);
	ihv1[3] += rotate_left(Q1[Qoffset+73],30);
	ihv1[4] += rotate_left(Q1[Qoffset+72],30);
	ihv2[0] += Q2[Qoffset+76];
	ihv2[1] += Q2[Qoffset+75];
	ihv2[2] += rotate_left(Q2[Qoffset+74],30);
	ihv2[3] += rotate_left(Q2[Qoffset+73],30);
	ihv2[4] += rotate_left(Q2[Qoffset+72],30);
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


void verifyQ56(const string& filename)
{
	ifstream ifs(filename.c_str(), ios::binary);
	if (!ifs)
	{
		cout << "Error: failed to open '" << filename << "'!" << endl;
		return;
	}
	ifs.seekg(0, ifs.end);
	size_t len = ifs.tellg();
	ifs.seekg(0, ifs.beg);

	len /= sizeof(q56sol_t);

	vector<q56sol_t> q56sols;
	q56sols.resize(len);

	ifs.read((char*)(&q56sols[0]), len*sizeof(q56sol_t));

	cout << "Loaded " << len << " Q56 solutions from '" << filename << "'." << endl;
	ifs.close();

	size_t okcnt = 0;
	for (size_t i = 0; i < q56sols.size(); ++i)
	{
		if (verifyQ56(q56sols[i]))
		{
			++okcnt;
		}
	}
	cout << "Verified: " << okcnt << " OK out of " << q56sols.size() << "." << endl;
}

void verifyQ56_new(const string& filename)
{
	ifstream ifs(filename.c_str(), ios::binary);
	if (!ifs)
	{
		cout << "Error: failed to open '" << filename << "'!" << endl;
		return;
	}
	ifs.seekg(0, ifs.end);
	size_t len = ifs.tellg();
	ifs.seekg(0, ifs.beg);

	len /= sizeof(q56sol_t);

	vector<q56sol_t> q56sols;
	q56sols.resize(len);

	ifs.read((char*)(&q56sols[0]), len*sizeof(q56sol_t));

	cout << "Loaded " << len << " Q56 solutions from '" << filename << "'." << endl;
	ifs.close();

	size_t okcnt = 0;
	for (size_t i = 0; i < q56sols.size(); ++i)
	{
		if (verifyQ56_new(q56sols[i]))
		{
			++okcnt;
		}
	}
	cout << "Verified: " << okcnt << " OK out of " << q56sols.size() << "." << endl;
}



///////////////////////////////////////////////////////////
//struct basesol_t {
//        uint32_t Q[6];  // Q12,..,Q17
//        uint32_t m[16]; // W6,...,W21
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
std::mutex basesols_mut;

void save_basesols(const string& filename)
{
	if (filename.empty()) 
	{
		return;
	}
	ofstream ofs((filename+".tmp").c_str(), ios::binary|ios::trunc);
	if (!ofs)
	{
		cerr << "Cannot open file '" << filename << "'!" << endl;
		return;
	}
	ofs.write((char*)(&basesols[0]),basesols.size()*sizeof(basesol_t));
	ofs.close();
	remove(filename.c_str());
	rename((filename+".tmp").c_str(), filename.c_str());
//	try {boost::filesystem::remove(filename);}catch(std::exception&e) { cout << e.what() << endl; }
//	try {boost::filesystem::rename(filename+".tmp", filename);}catch(std::exception&e) {cout << e.what() << endl; }
	cout << "Written " << basesols.size() << " basesols to '" << filename << "'." << endl;
}
vector<basesol_t> load_basesols(const string& filename)
{
	std::vector<basesol_t> ret;
	ifstream ifs(filename.c_str(), ios::binary);
	if (!!ifs)
	{
		ifs.seekg(0, ifs.end);
		size_t len = ifs.tellg();
		ifs.seekg(0, ifs.beg);

		len /= sizeof(basesol_t);
		ret.resize(len);

		ifs.read((char*)(&ret[0]), len*sizeof(basesol_t));

		cout << "Loaded " << len << " basesols from '" << filename << "'." << endl;

		for (size_t i = 0; i < ret.size(); )
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

	}
	return ret;
}
void load_basesols(const vector<string>& inputfile)
{
	// lock locks mutex untils it is destroyed upon function exit
	std::lock_guard<std::mutex> lock(basesols_mut);
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
		cout << "Loaded " << basesols.size() << " basesols from inputfile(s)." << endl;
	}
}
// run in a separate thread by main()
void process_basesol()
{
	cout << "Processing basesols to disk in the background" << endl;
	while (true)
	{
		// update results on disk every minute
		std::this_thread::sleep_for(std::chrono::seconds(120));
		{
			// lock locks mutex untils it is destroyed upon function exit
			std::lock_guard<std::mutex> lock(basesols_mut);

//			sort(basesols.begin(), basesols.end());
//			basesols.erase( unique(basesols.begin(),basesols.end()), basesols.end());

			save_basesols(outputfile);
		}
	}

}

/*extern*/ bool BASESOL_OK = true;
void found_basesol(uint32_t main_m1[80], uint32_t main_Q1[85], unsigned mainblockoffset)
{
	// lock locks mutex untils it is destroyed upon function exit
	std::lock_guard<std::mutex> lock(basesols_mut);

	basesol_t basesol;

	for (int t = 12; t <= 17; ++t)
		basesol.Q[t-12] = main_Q1[4+t];

	// compute ALL W0,...,W79
	sha1_me_generalised(main_m1, mainblockoffset);

	for (int t = 6; t < 6+16; ++t)
		basesol.m[t-6] = main_m1[t];

	BASESOL_OK = verify(basesol);
	if (BASESOL_OK)
	{
		basesols.push_back(basesol);
		if (hashclash::hw(basesols.size()) == 1)
		{
			cout << "{" << basesols.size() << "}" << flush;
		}
		if (max_basesols > 0 && basesols.size() == max_basesols)
		{
			if (!outputfile.empty())
				save_basesols(outputfile);
			exit(0);
		}
	}
}




int main(int argc, char** argv)
{

	timer::timer runtime;
	try 
	{
		std::string seedstr;
		const char* seedchars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
		for (int i = 0; i < 22; ++i)
			seedstr += seedchars[xrng128() % 62];
		max_basesols = 0;

		po::options_description
			opt_cmds("Allowed commands"),
			opt_opts("Allowed options"),
			all("Allowed options");
		opt_cmds.add_options()
			("help,h",        "Show options\n")
			("genbasesol,g",  "Find and store base solutions")
			;
		if (compiled_with_cuda())
		    opt_cmds.add_options()
			("query,q",       "Query CUDA devices")
			("sha1bench,b",   "Benchmark SHA-1 on CUDA device")
			("cudaattack,a",  "Load base solutions and do neutral bits in CUDA")
			;
		opt_cmds.add_options()
//			("cpuattack,c",     "Load base solutions and do neutral bits on CPU")
			("verifyQ56,v",     "Verify Q56 solutions") // new: stored as Q56-state
//			("verifyQ56old",     "Verify Q56 solutions (Q36-state)") // old: stored as Q36-state
			;
		opt_opts.add_options()
			("seed,s"
				, po::value<string>(&seedstr)
				, "Set SEED value")
			("inputfile,i"
				, po::value<vector< string> >(&inputfile)
				, "Inputfile(s) (may be given more than once)")
			("outputfile,o"
				, po::value<string>(&outputfile)
				, "Outputfile")
			("disablebackwardsfilter"
				, "Disable backwards error prob. check")
			("maxbasesols,m"
				, po::value<int>(&max_basesols)
				, "Stop when # basesols have been generated")
			;
		if (compiled_with_cuda())
		    opt_opts.add_options()
			("cudadevice,d"
				, po::value<int>(&cuda_device)->default_value(0)
				, "Set CUDA device")
			("cudascheduler"
				, po::value<int>(&cuda_scheduler)->default_value(3)
				, "0=auto, 1=spin, 2=yield, 3=blockingsync")
			("cudablocks"
				, po::value<int>(&cuda_blocks)->default_value(-1)
				, "Set # blocks to start on GPU")
			("cudathreadsperblock"
				, po::value<int>(&cuda_threads_per_block)->default_value(-1)
				, "Set # threads per block to start on GPU")
			;
		all.add(opt_cmds).add(opt_opts);
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, all), vm);
		{
			std::ifstream ifs("config.cfg");
			if (ifs) po::store(po::parse_config_file(ifs,all), vm);
		}
		po::notify(vm);
		disable_backwards_filter = vm.count("disablebackwardsfilter");


		if (vm.count("help") 
			|| (vm.count("query")==0 && vm.count("sha1bench")==0 && vm.count("genbasesol")==0 
				&& vm.count("cudaattack")==0 && vm.count("cpuattack")==0 
				&& vm.count("verifyQ56")==0 && vm.count("verifyQ56old")==0))
		{
			cout << opt_cmds << opt_opts << endl;
			return 0;
		}
		// process seed str
		cout << "Current seed string: " << seedstr << endl;
		seed(0);
		for (int i = 0; i < seedstr.size(); ++i)
		{
			uint32_t x = uint32_t(seedstr[i]);
			x = rotate_right( (x<<24)|(x<<16)|(x<<8)|x, (i&31) );
			addseed(x ^ i);
		}

		if (vm.count("query"))
		{
			cuda_query();
		}
		if (vm.count("sha1bench"))
		{
			gpusha1benchmark();
		}
		if (vm.count("genbasesol"))
		{
			load_basesols(inputfile);
			if (!basesols.empty())
			{
				save_basesols(outputfile);
			}
			else
			{
				if (vm.count("outputfile"))
				{
					// start a separate thread to process and store basesols to disk
					std::thread tmp(process_basesol);
					// start the attack in the main thread
					start_attack(); // never returns
				}
				else
				{
					cout << "Warning: no outputfile given, results will be lost." << endl;
					// start the attack in the main thread
					start_attack(); // never returns
				}
			}
		}
		if (vm.count("cudaattack"))
		{
			if (vm.count("inputfile"))
			{
				load_basesols(inputfile);
				cuda_main(basesols);
			}
			else
			{
				cout << "Error: no inputfile with basesolutions given!" << endl; 
			}
		}
		if (vm.count("cpuattack"))
		{
			if (vm.count("inputfile"))
			{
				load_basesols(inputfile);
				cpu_main(basesols);
			}
			else
			{
				cout << "Error: no inputfile with basesolutions given!" << endl; 
			}
		}
		if (vm.count("verifyQ56"))
		{
			if (vm.count("inputfile"))
			{
				for (size_t i = 0; i < inputfile.size(); ++i)
				{
					verifyQ56_new(inputfile[i]);
				}
			}
			else
			{
				cout << "Error: no inputfile with Q56 solutions given!" << endl;
			}
		}
		if (vm.count("verifyQ56old"))
		{
			if (vm.count("inputfile"))
			{
				for (size_t i = 0; i < inputfile.size(); ++i)
				{
					verifyQ56(inputfile[i]);
				}
			}
			else
			{
				cout << "Error: no inputfile with Q56 solutions given!" << endl;
			}
		}
	}
	catch (std::exception& e)
	{
		cerr << "Caught exception:" << endl << e.what() << endl;
	}
	cout << "Runtime: " << runtime.time() << endl;
	return 0;
}
