'''
Example usage:
python removeInstances.py final.arff 1 2 3
This command will filter instance numbers 1,2 and 3 and write new file at instanceFiltered.arff 
'''

import numpy as np
import sys


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    raise ValueError("No such file exists.")  

f = open(filename,"r")
filtered = open("instanceFiltered.arff","w")

instances_to_remove = []

for i in xrange(len(sys.argv[2:])):
	num = int(sys.argv[2:][i])
	instances_to_remove.append(num)

a = f.readline()
cnt = 1
while a:
	if not(cnt in instances_to_remove):
		filtered.write(a)
	cnt += 1
	a = f.readline()

f.close()
filtered.close()
