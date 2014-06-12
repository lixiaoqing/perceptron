#!/usr/bin/python

import sys
coding = 'gbk'
fin = sys.argv[1]
f = open('train.conll','w')
for s in open(fin):
	s = s.decode(coding)
	for w in s.split():
		if len(w) == 1:
			print >>f, '{}\t{}'.format(w.encode(coding),'S')
		else:
			print >>f, '{}\t{}'.format(w[0].encode(coding),'B')
			for c in w[1:-1]:
				print >>f, '{}\t{}'.format(c.encode(coding),'M')
			print >>f, '{}\t{}'.format(w[-1].encode(coding),'E')
	print >>f
f.close()

ftok = open('tagset_for_token','w')
print >>ftok,'unk B M E S'
ftok.close()
flasttag = open('tagset_for_last_tag','w')
print >>flasttag,'unk B M E S'
print >>flasttag,'B M E'
print >>flasttag,'M M E'
print >>flasttag,'E S B'
print >>flasttag,'S S B'
flasttag.close()
