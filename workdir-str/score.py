#!/usr/bin/python
def ws2ts(ws):
	ts = ''
	for w in ws:
		if len(w) == 1:
			ts += 'S'
		else:
			ts += 'B'
			for c in w[1:-1]:
				ts += 'M'
			ts += 'E'
	return ts

#fns = ['pku','as','cityu','msr']
fns = ['pku']
for fn in fns:
	gold,accurate,out = 0,0,0
	iv_gold,oov_gold,iv_acc,oov_acc = 0,0,0,0
	f = open(fn+'_training_words.txt')
	iv = set(f.read().decode('gbk').split())
	f.close()
	f1 = open(fn+'_out')
	f2 = open(fn+'_test_gold.txt')
	for so,sg in zip(f1,f2):
		wo = so.decode('gbk').split()
		to = ws2ts(wo)
		wg = sg.decode('gbk').split()
		tg = ws2ts(wg)
		gold += len(wg)
		out += len(wo)
		pos = 0
		for w in wg:
			if w in iv:
				iv_gold += 1
				if to[pos:pos+len(w)] == tg[pos:pos+len(w)]:
					accurate += 1
					iv_acc += 1
			else:
				oov_gold += 1
				if to[pos:pos+len(w)] == tg[pos:pos+len(w)]:
					accurate += 1
					oov_acc += 1
			pos += len(w)
	f1.close()
	f2.close()
	r = float(accurate)/gold
	p = float(accurate)/out
	f = 2*p*r/(p+r)
	riv = float(iv_acc)/iv_gold
	roov = float(oov_acc)/oov_gold
	print fn
	print 'r\tp\tF\troov\triv'
	print '{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(r,p,f,roov,riv)
