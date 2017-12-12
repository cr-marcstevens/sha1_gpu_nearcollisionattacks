/*****
  Copyright (C) 2015 Pierre Karpman, INRIA France/Nanyang Technological University Singapore (-2016), CWI (2016/2017), L'Universite Grenoble Alpes (2017-)
            (C) 2015 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.

  This file is part of sha1_gpu_nearcollisionattacks source-code and released under the MIT License
*****/

#include "main.hpp"
#include "neutral_bits_packing.hpp"

#include "sha1detail.hpp"
#include "rng.hpp"

#include <timer.hpp>
#include <program_options.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <set>

#include "tables_org.hpp"

using namespace std;
using namespace hashclash;
namespace po = program_options;

int cuda_device, cuda_blocks, cuda_threads_per_block, cuda_scheduler;
vector<string> inputfile;
string outputfile;
int max_basesols;

vector<q13sol_t> load_q13sols(const string& filename, int linenr = -1)
{
	std::vector<q13sol_t> ret;
	ifstream ifs(filename.c_str());
	if (!ifs)
		return ret;
	unsigned curline = 0;
	while (!!ifs)
	{
		std::string line;
		getline(ifs, line);
		if (linenr >= 0 && curline++ != linenr)
			continue;

		size_t pos = line.find("B!");
		while (pos < line.size())
		{
			size_t pos2 = line.find(" ", pos);
			ret.push_back(decode_q13sol(line.substr(pos, pos2 - pos)));
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

void verifyQ61sols(const std::string& filename)
{
	using namespace tbl_org;
	std::set<q61sol_t> Q53sols, Q61sols, Q80sols;
	
	ifstream ifs(filename.c_str());
	if (!ifs)
	{
		std::cout << "Failed to open file: " << filename << std::endl;
		return;
	}
	size_t solcount = 0;
	while (!!ifs)
	{
		std::string line;
		getline(ifs, line);
		
		if (line.substr(0,2) != "Q!")
			continue;
		q61sol_t sol = decode_q61sol(line);
		++solcount;
		
		uint32_t m1[80];
		uint32_t m2[80];
		uint32_t Q1[85];
		uint32_t Q2[85];
		
		bool Q53sol = true, Q61sol = true, Q80sol = true;
		
		for (unsigned i = 0; i < 16; ++i)
		{
			m1[i] = sol.m[i];
			m2[i] = m1[i] ^ DV_DW[i];
		}
		for (int i = -4; i <= 0; ++i)
		{
			Q1[4+i] = Qset1mask[4+i];
			Q2[4+i] = Q1[4+i] + dQ[4+i];
		}
		sha1_me_generalised(m1,0);
		sha1_me_generalised(m2,0);
		for (int i = 0; i < 80; ++i)
		{
			sha1_step(i,Q1,m1);
			sha1_step(i,Q2,m2);
//			std::cout << "dQ" << (i+1) << " = " << std::hex << Q2[4+i+1]-Q1[4+i+1] << std::dec << std::endl;
		}
		
		for (int i = 49; i <= 53; ++i)
			if (Q2[4+i] != Q1[4+i])
				Q53sol = false;
		if (Q53sol)
			Q53sols.insert(sol);
			
		for (int i = 57; i <= 61; ++i)
			if (Q2[4+i] != Q1[4+i])
				Q61sol = false;
		if (Q61sol)
			Q61sols.insert(sol);
		
		uint32_t CV1[5]={
			Q1[4+0]+Q1[4+80]
			, Q1[4-1]+Q1[4+79]
			,rotate_left(Q1[4-2],30)+rotate_left(Q1[4+78],30)
			,rotate_left(Q1[4-3],30)+rotate_left(Q1[4+77],30)
			,rotate_left(Q1[4-4],30)+rotate_left(Q1[4+76],30)
			};
		uint32_t CV2[5]={
			Q2[4+0]+Q2[4+80]
			, Q2[4-1]+Q2[4+79]
			,rotate_left(Q2[4-2],30)+rotate_left(Q2[4+78],30)
			,rotate_left(Q2[4-3],30)+rotate_left(Q2[4+77],30)
			,rotate_left(Q2[4-4],30)+rotate_left(Q2[4+76],30)
			};
		for (int i = 0; i < 5; ++i)
			if (CV1[i]!=CV2[i])
				Q80sol = false;
				
		if (Q80sol)
		{
			Q80sols.insert(sol);
			std::cout << "Collision found!" << std::endl;
			std::cout << "m1 = {";
			for (int i = 0; i<16; ++i)
				printf(" %08x", m1[i]);
			std::cout << " };\n";
			std::cout << "m2 = {";
			for (int i = 0; i<16; ++i)
				printf(" %08x", m2[i]);
			std::cout << " };\n";
		}
	}
	std::cout << "Loaded " << solcount << " solutions:" << std::endl;
	std::cout << "Number of Q53sols: " << Q53sols.size() << std::endl;
	std::cout << "Number of Q61sols: " << Q61sols.size() << std::endl;
	std::cout << "Number of Q80sols: " << Q80sols.size() << std::endl;
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
		int inputlinenr = -1;

		po::options_description
			opt_cmds("Allowed commands"),
			opt_opts("Allowed options"),
			all("Allowed options");
		opt_cmds.add_options()
			("help,h", "Show options\n")
			("genbasesol,g", "Find and store base solutions")
			;
		if (compiled_with_cuda())
		    opt_cmds.add_options()
			("query,q", "Query CUDA devices")
			("sha1bench,b", "Benchmark SHA-1 on CUDA device")
			("cudaattack,a", "Load base solutions and perform GPU attack part")
			;
		opt_cmds.add_options()
			("verifyQ61,v", "Verify Q61 solutions")
			;

		opt_opts.add_options()
			("seed,s"
				, po::value<std::string>(&seedstr)
				, "Set SEED string")
			("inputfile,i"
				, po::value<std::vector<std::string>>(&inputfile)
				, "Inputfile(s) (may be given more than once)")
			("inputline,l"
				, po::value<int>(&inputlinenr)
				, "Only load basesols from this linenr (0=1st line)")
			("outputfile,o"
				, po::value<string>(&outputfile)
				, "Outputfile")
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
				, "Set # block to start on GPU")
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

		if (vm.count("help") 
			|| (vm.count("query")==0 && vm.count("sha1bench")==0 && vm.count("genbasesol")==0
				&& vm.count("cudaattack")==0 && vm.count("verifyQ61")==0))
		{
			cout << opt_cmds << opt_opts << endl;
			return 0;
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
		if (vm.count("query"))
		{
			cuda_query();
		}
		if (vm.count("sha1bench"))
		{
			gpusha1benchmark();
		}
		if (vm.count("cudaattack"))
		{
			if (!inputfile.empty())
			{
				q13sols = load_q13sols(inputfile.back(), inputlinenr);
				q14sols.clear();
				for (size_t i = 0; i < q13sols.size(); ++i)
				{
					verify(q13sols[i]);
					step13nb(q13sols[i]);
				}
				std::cout << "Generated " << q14sols.size() << " Q14-solutions from " << q13sols.size() << " Q13-solutions." << std::endl;
				cuda_main(q14sols);
			}
			else
			{
				cout << "Error: no inputfile with basesolutions given!" << endl; 
			}
		}
		if (vm.count("genbasesol"))
		{
			if (outputfile.empty())
			{
				cout << "Warning: no outputfile given, results will be lost." << endl;
			}
			gen_q13sols(); // never returns, uses exit(0)
		}
		if (vm.count("verifyQ61"))
		{
			if (!inputfile.empty())
			{
				for (size_t i = 0; i < inputfile.size(); ++i)
				{
					verifyQ61sols(inputfile[i]);
				}
			}
			else
			{
				cout << "Error: no inputfile with Q61 solutions given!" << endl;
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
