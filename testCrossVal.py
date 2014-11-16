import numpy as np
import sys
from CrossValidation import CrossValidation
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

if len(sys.argv) > 1:
    filename = 'data/' + sys.argv[1]
else:
    filename = "data/svmData.dat"   
    
print sys.argv[1]

allData = np.loadtxt(filename, delimiter=',')
Xtrain = allData[:, :-1]
Ytrain = allData[:, -1] 
n_features = Xtrain.shape[1]
myModel = linear_model.LogisticRegression()
clf = RandomForestClassifier(n_estimators=10)
myModel = svm.SVC(kernel = 'linear', C = 1)

# clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

# scores = CrossValidation().cross_val_score(clf, Xtrain, Ytrain, cv=10)
# #print scores
# print "RandomForestClassifier : ", scores.mean()

# clf = ExtraTreesClassifier(n_estimators=120, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

# scores = CrossValidation().cross_val_score(clf, Xtrain, Ytrain, cv=10)
# print "ExtraTreesClassifier : ", scores.mean()

# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400, algorithm="SAMME.R")

# scores = CrossValidation().cross_val_score(clf, Xtrain, Ytrain, cv=10)
# print "AdaBoostClassifier : ", scores.mean()

clf = linear_model.LogisticRegression()

scores = CrossValidation(Xtrain, Ytrain, cv=10).cross_val_score(clf)
print "LogisticRegression : ", scores.mean()

# M witclf = svm.SVC(kernel = 'linear', C = 0.01)

# scores = CrossValidation().cross_val_score(clf, Xtrain, Ytrain, cv=10)
# print "SVM with linear kernel (C = 0.01): ", scores.mean()

# clf = svm.SVC(kernel='rbf', gamma = 1, C = 50)

# scores = CrossValidation().cross_val_score(clf, Xtrain, Ytrain, cv=10)
# print "SVh gaussian kernel (C = 50, gamma = 1): ", scores.mean()

print ""


