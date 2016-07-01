

from os import listdir
from os.path import isfile, join

mypath = "/auto/rcf-40/al_227/nlg-05/gits/seq2seq-attn/predictions/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith("out_")]

references = []
roc = "/home/rcf-40/al_227/nlg-05/DATA/tasks/task2/roc/test_T_c_1.txt"
with open(roc,"r") as f:
	for line in f:
		references.append(line.strip())


for onlyfile in onlyfiles:
	good = []
	bad = []
	i = 0
	with open(join(mypath, onlyfile),"r") as f:
		for line in f:
			line = line.strip()
			if not line.startswith("PRED SCORE"):
				continue
			s = line.split(" ")[5]
		
			r = references[i]
			l = len(r.split())
		
			if i%2==0:
				good.append((float(s),r,l))
			else:
				bad.append((float(s),r,l))
			i+=1


	corr_norm = 0.0
	corr = 0.0
	all = 0.0
	conf = 0
	for x,y in zip(good,bad):
		all +=1
		if x[0]/x[2] >= y[0]/y[2]:
			corr_norm+=1
		if x[0] >= y[0]:
			corr+=1



	print onlyfile, corr/all,corr_norm/all
