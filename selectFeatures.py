'''
Example usage:
python selectFeatures.py final.arff 0 1 2
This command will filter features at index 0 1 and 2 and write new file at filtered.arff 
'''

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

while a:
	# to remove new line character
	a = a[0:len(a)-1]
	b = a.split(",")

	for i in xrange(len(b)):
		if (i == len(b)-1):
			filtered.write(b[i])
			filtered.write("\n")
		elif i in selected_features:
			filtered.write(b[i])
			filtered.write(",")
			

	a = f.readline()

f.close()
filtered.close()