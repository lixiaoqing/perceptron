#include "stdafx.h"

void TrimLine(string &line);
int s2i(string &s);
double s2d(string &s);
string i2s(int i);
void Split(vector<string> &vs, string &s);
void Split(vector<string> &vs, string &s, string &sep);

size_t fnv1_hash(const vector<int> &ivec);
size_t djb2_hash(const vector<int> &ivec);
size_t bkdr_hash(const vector<int> &ivec);
