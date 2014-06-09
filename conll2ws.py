#!/usr/bin/python

import sys
coding = 'gbk'
id2char = {}
for s in open('char2id'):
	s=s.split()
	id2char[s[1]] = s[0]

f = open('pku_out','w')
for s in open('output'):
	if s.strip() == '':
		f.write('\n')
		continue
	s = s.split()
	if s[1] in '14':
		f.write(id2char.get(s[0],'x')+' ')
	else:
		f.write(id2char.get(s[0],'x'))
f.close()

