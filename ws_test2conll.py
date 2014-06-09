#!/usr/bin/python

import sys
coding = 'gbk'
char2id = {}
for s in open('char2id'):
	s=s.decode(coding).split()
	char2id[s[0]] = s[1]

fin = sys.argv[1]
f = open('test.conll','w')
for s in open(fin):
	s = s.decode(coding).strip()
	for c in s:
		print >>f,char2id.get(c,-1)
	print >>f
f.close()

