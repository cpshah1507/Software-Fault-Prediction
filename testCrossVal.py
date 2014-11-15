import numpy as np
import sys
from CrossValidation import CrossValidation
from sklearn import linear_model

if len(sys.argv) > 1:
    filename = 'data/' + sys.argv[1]
else:
    filename = "data/svmData.dat"   
    
print sys.argv[1]

allData = np.loadtxt(filename, delimiter=',')
Xtrain = allData[:, :-1]
Ytrain = allData[:, -1] 
myModel = linear_model.LogisticRegression()
scores = CrossValidation().cross_val_score(myModel, Xtrain, Ytrain, cv=10)
print scores
print scores.mean()
