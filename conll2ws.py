#!/usr/bin/python

import sys
coding = 'gbk'
f = open('pku_out','w')
for s in open('output'):
	if s.strip() == '':
		f.write('\n')
		continue
	s = s.split()
	if s[1] in 'SE':
		f.write(s[0]+' ')
	else:
		f.write([0])
f.close()

