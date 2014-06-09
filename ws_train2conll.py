#!/usr/bin/python

import sys
coding = 'gbk'
char2id = {}
tag2id = {'S':'1','B':'2','M':'3','E':'4'}
charid = 1
fin = sys.argv[1]
f = open('train.conll','w')
for s in open(fin):
	s = s.decode(coding)
	for w in s.split():

		for c in w:
			if c not in char2id:
				char2id[c] = charid
				charid += 1
		if len(w) == 1:
			print >>f, '{}\t{}'.format(char2id[w],tag2id['S'])
		else:
			print >>f, '{}\t{}'.format(char2id[w[0]],tag2id['B'])
			for c in w[1:-1]:
				print >>f, '{}\t{}'.format(char2id[c],tag2id['M'])
			print >>f, '{}\t{}'.format(char2id[w[-1]],tag2id['E'])
	print >>f
f.close()

fchar2id = open('char2id','w')
for k,v in sorted(char2id.items(),key=lambda d:d[1]):
	print >>fchar2id,'{}\t{}'.format(k.encode(coding),v)
fchar2id.close()
ftag2id = open('tag2id','w')
for k,v in tag2id.items():
	print >>ftag2id,'{}\t{}'.format(k.encode(coding),v)
ftag2id.close()
ftok = open('tagset_for_token','w')
print >>ftok,'-1 1 2 3 4'
ftok.close()
flasttag = open('tagset_for_last_tag','w')
print >>flasttag,'-1 1 2 3 4'
print >>flasttag,'0 1 2'
print >>flasttag,'1 1 2'
print >>flasttag,'2 3 4'
print >>flasttag,'3 3 4'
print >>flasttag,'4 1 2'
flasttag.close()
