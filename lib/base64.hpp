/******************************************************************************************\
*
*    Copyright (C) 2016 Marc Stevens, Centrum Wiskunde & Informatica (CWI), Amsterdam.
*
*    This file is released under the MIT License, see https://opensource.org/licenses/MIT
*    
\******************************************************************************************/

#ifndef BASE64_HPP
#define BASE64_HPP

#include <string>
#include <vector>

static inline std::string base64_encode(const std::string& in);
static inline std::string base64_decode(const std::string& in);

namespace detail
{

	// template class so that instantiations in different compilation units are merged
	template<int dummy>
	struct base64_helper
	{
		static const char* chars;
		static std::vector<int> LUT;

		static std::string base64_encode(const std::string& in)
		{
			std::string out;
			out.reserve( ((in.size()+2)/3)*4 );

			int val=0, bits=-6;
			for (std::size_t i = 0; i < in.size(); ++i)
			{
				val = (val<<8) + (int)((unsigned char)(in[i]));
				for (bits += 8; bits >= 0; bits -= 6)
				{
					out.push_back( chars[ (val>>bits)&0x3F ] );
				}
			}

			if (bits > - 6)
			{
				val <<= 8;
				bits += 8;
				out.push_back( chars[ (val>>bits)&0x3F ] );
			}
			while (out.size()%4)
			{
				out.push_back('=');
			}
			return out;
		}

		static std::string base64_decode(const std::string& in)
		{
			std::string out;
			out.reserve((in.size()/4)*3);
			
			if (LUT.empty())
			{
				LUT.resize(256,-1);
				for (int i = 0; i < 64; ++i)
				{
					LUT[(int)( (unsigned char)(chars[i]) )] = i;
				}
			}

			int val=0, bits=-8;
			for (std::size_t i = 0; i < in.size(); ++i)
			{
				int c = LUT[(int)( (unsigned char)(in[i]) )];
				if (c == -1)
				{
					break;
				}

				val = (val << 6) + c;
				bits += 6;
				if (bits >= 0)
				{
					out.push_back( (char)( (val>>bits)&0xFF ) );
					bits -= 8;
				}
			}

			return out;
		}
	};
	
	template<int dummy>
	const char* base64_helper<dummy>::chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	template<int dummy>
	std::vector<int> base64_helper<dummy>::LUT;

}

static inline std::string base64_encode(const std::string& in)
{
	return detail::base64_helper<0>::base64_encode(in);
}

static inline std::string base64_decode(const std::string& in)
{
	return detail::base64_helper<0>::base64_decode(in);
}

#endif // BASE64_HPP
