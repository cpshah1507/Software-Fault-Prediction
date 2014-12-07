from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    raise ValueError("No such file exists.")  
    
print sys.argv[1]
j = int(sys.argv[2])

allData = np.loadtxt(filename, delimiter=',')
Xtrain = allData[:, :-1]
Ytrain = allData[:, -1]

n_featuresOld = Xtrain.shape[1]
XtrainOld = Xtrain
for i in range(j):
	print 'Old # features = ', Xtrain.shape[1]

	clf = ExtraTreesClassifier()
	Xtrain = clf.fit(Xtrain, Ytrain).transform(Xtrain)
        if(i==0):
		arr = clf.feature_importances_
		arrSum = np.sum(arr)

	print 'New # features = ', Xtrain.shape[1]

featureContribution = {}
indices = np.sort(np.unique(np.where(XtrainOld[0]==x)[0][0] for x in Xtrain[0])).tolist()
print 'Features selected are', indices
print 'Weights of important features are: '
for i in range(len(indices)):
	featureContribution[indices[i]] = (arr[indices[i]]/arrSum)*100
print featureContribution

