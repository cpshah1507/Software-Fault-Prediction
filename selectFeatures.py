import numpy as np
import sys


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    raise ValueError("No such file exists.")  

f = open(filename,"r")
filtered = open("filtered.arff","w")

selected_features = []

for i in xrange(len(sys.argv[2:])):
	num = int(sys.argv[2:][i])
	selected_features.append(num)

a = f.readline()
cnt = 1
while cnt<10:
#while a:
	# to remove new line character
	a = a[0:len(a)-1]
	b = a.split(",")

	for i in xrange(len(b)):
		if not(i in selected_features):
			filtered.write(b[i])
			if(i < len(b)-1):
				filtered.write(",")
			else:
				filtered.write("\n")

	a = f.readline()
	cnt += 1

f.close()
filtered.close()