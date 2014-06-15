#include "myutils.h"

void Split(vector <string> &vs, string &s)
{
	vs.clear();
	stringstream ss;
	string e;
	ss << s;
	while(ss >> e)
		vs.push_back(e);
}

void Split(vector <string> &vs, string &s, string &sep)
{
	int cur = 0,next;
	next = s.find(sep);
	while(next != string::npos)
	{
		if(s.substr(cur,next-cur) !="")
			vs.push_back(s.substr(cur,next-cur));
		cur = next+sep.size();
		next = s.find(sep,cur);
	}
	vs.push_back(s.substr(cur));
}

void TrimLine(string &line)
{
	line.erase(0,line.find_first_not_of(" \t\r\n"));
	line.erase(line.find_last_not_of(" \t\r\n")+1);
}

int s2i(string &s)
{
	int i;
	stringstream ss;
	ss<<s;
	ss>>i;
	return i;
}

double s2d(string &s)
{
	double d;
	stringstream ss;
	ss<<s;
	ss>>d;
	return d;
}

string i2s(int i)
{
	stringstream ss;
	ss<<i;
	return ss.str();
}

size_t fnv1_hash(const vector<int> &ivec)
{
	size_t hash = 2166136261;
	for (auto &i : ivec)
	{
		hash *= 16777619;
		hash ^= i;
	}
	return hash;
}

size_t djb2_hash(const vector<int> &ivec)
{
	size_t hash = 5381;

	for (auto &i : ivec)
	{
		hash = ((hash << 5) + hash) + i; /* hash * 33 + c */
	}
	return hash;
}

size_t bkdr_hash(const vector<int> &ivec)
{
	//size_t seed = 131; // 31 131 1313 13131 131313 etc..
	size_t hash = 0;

	for (auto &i : ivec)
	{
		hash = hash * 131 + i;
	}
	return (hash & 0x7FFFFFFF);
}

