# Combine CM1, MW1, PC1, PC3 and PC4
f1 = open("CM1.arff","r")
f2 = open("MW1.arff","r")
f3 = open("PC1.arff","r")
f4 = open("PC3.arff","r")
f5 = open("PC4.arff","r")

c1 = open("file1.arff","w")

data = f1.read()
c1.write(data)
c1.write('\n')

data = f2.read()
c1.write(data)
c1.write('\n')

data = f3.read()
c1.write(data)
c1.write('\n')

data = f4.read()
c1.write(data)
c1.write('\n')

data = f5.read()
c1.write(data)
c1.write('\n')

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()

c1.close()

# Remove Feature of Decision_Density

f1 = open("file1.arff","r")
f2 = open("file2.arff","w")

while 1:
	dataline = f1.readline()
	if len(dataline) == 0:
		break
	dataline = dataline[:-1] # removing new line character
	dataline = dataline.split(",") # get all features and label to list
	dataline1 = dataline[0:9]+dataline[10:] # code to remove decision_density feature (it's at 10th position)
	f2.write(",".join(dataline1))
	f2.write('\n')

f1.close()
f2.close()

# Combine KC3 and MC2

f1 = open("KC3.arff","r")
f2 = open("MC2.arff","r")

c1 = open("file3.arff","w")

data = f1.read()
c1.write(data)
c1.write('\n')

data = f2.read()
c1.write(data)
c1.write('\n')

f1.close()
f2.close()
c1.close()

# Remove decision_density from that

f1 = open("file3.arff","r")
f2 = open("file4.arff","w")

while 1:
	dataline = f1.readline()
	if len(dataline) == 0:
		break
	dataline = dataline[:-1] # removing new line character
	dataline = dataline.split(",") # get all features and label to list
	dataline1 = dataline[0:9]+dataline[10:] # code to remove decision_density feature (it's at 10th position)
	f2.write(",".join(dataline1))
	f2.write('\n')

f1.close()
f2.close()

# Add MC1 to file4.arff

f1 = open("MC1.arff","r")
c1 = open("file4.arff","w")

data = f1.read()
c1.write(data)
c1.write('\n')

f1.close()
c1.close()


# Remove global_data_complexity and global_data_density

f1 = open("file4.arff","r")
f2 = open("file5.arff","w")

while 1:
	dataline = f1.readline()
	if len(dataline) == 0:
		break
	dataline = dataline[:-1] # removing new line character
	dataline = dataline.split(",") # get all features and label to list
	dataline1 = dataline[0:16]+dataline[18:]
	f2.write(",".join(dataline1))
	f2.write('\n')

f1.close()
f2.close()

# Combine file2 and file5

f1 = open("file2.arff","r")
f2 = open("file5.arff","r")
f3 = open("PC5.arff","r")

c1 = open("file6.arff","w")

data = f1.read()
c1.write(data)

data = f2.read()
c1.write(data)

data = f3.read()
c1.write(data)
c1.write('\n')

f1.close()
f2.close()
f3.close()
c1.close()



