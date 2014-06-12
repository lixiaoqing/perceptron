#!/usr/bin/python

import sys
coding = 'gbk'
fin = sys.argv[1]
f = open('test.conll','w')
for s in open(fin):
	s = s.decode(coding).strip()
	for c in s:
		print >>f,c.encode(coding)
	print >>f
f.close()

