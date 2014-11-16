import numpy as np
import sys
from CrossValidation import CrossValidation
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from Smote import oversampleData

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    raise ValueError("No such file exists.")  
    
print sys.argv[1]

allData = np.loadtxt(filename, delimiter=',')
Xtrain = allData[:, :-1]
Ytrain = allData[:, -1] 
n_features = Xtrain.shape[1]

#Xtrain, Ytrain = oversampleData(Xtrain, Ytrain, 60, 3)
crossVal = CrossValidation(Xtrain, Ytrain, cv=10)

print ""
numPos, numNeg = crossVal.instancesPerLabel()
print "Positive instances : ", numPos
print "Negative instances : ", numNeg
print ""
print "Accuracy Scores"
print ""

clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

scores = crossVal.cross_val_accuracy(clf)

print "RandomForestClassifier : ", scores.mean()

# clf = ExtraTreesClassifier(n_estimators=120, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

# scores = crossVal.cross_val_accuracy(clf)
# print "ExtraTreesClassifier : ", scores.mean()

# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400, algorithm="SAMME.R")

# scores = crossVal.cross_val_accuracy(clf)
# print "AdaBoostClassifier : ", scores.mean()

clf = linear_model.LogisticRegression()

scores = crossVal.cross_val_accuracy(clf)
print "LogisticRegression : ", scores.mean()

# clf = MultinomialNB()

# scores = crossVal.cross_val_accuracy(clf)
# print "MultinomialNB : ", scores.mean()

# clf = svm.SVC(kernel = 'linear', C = 0.01, class_weight = 'auto')

# scores = crossVal.cross_val_accuracy(clf)
# print "SVM with linear kernel (C = 0.01): ", scores.mean()

# clf = svm.SVC(kernel='rbf', gamma = 1, C = 50, class_weight = 'auto')

# scores = crossVal.cross_val_accuracy(clf)
# print "SVM gaussian kernel (C = 50, gamma = 1): ", scores.mean()

print ""
print "Precision Scores"
print ""

clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

scores = crossVal.cross_val_precision(clf)

#print scores
print "RandomForestClassifier : ", scores.mean()

# clf = ExtraTreesClassifier(n_estimators=120, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

# scores = crossVal.cross_val_precision(clf)
# print "ExtraTreesClassifier : ", scores.mean()

# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400, algorithm="SAMME.R")

# scores = crossVal.cross_val_precision(clf)
# print "AdaBoostClassifier : ", scores.mean()

clf = linear_model.LogisticRegression()

scores = crossVal.cross_val_precision(clf)
print "LogisticRegression : ", scores.mean()

# clf = MultinomialNB()

# scores = crossVal.cross_val_precision(clf)
# print "MultinomialNB : ", scores.mean()

# clf = svm.SVC(kernel = 'linear', C = 0.01)

# scores = crossVal.cross_val_precision(clf)
# print "SVM with linear kernel (C = 0.01): ", scores.mean()

# clf = svm.SVC(kernel='rbf', gamma = 1, C = 50)

# scores = crossVal.cross_val_precision(clf)
# print "SVM gaussian kernel (C = 50, gamma = 1): ", scores.mean()

print ""
print "Recall Scores"
print ""

clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

scores = crossVal.cross_val_recall(clf)

#print scores
print "RandomForestClassifier : ", scores.mean()

# clf = ExtraTreesClassifier(n_estimators=120, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

# scores = crossVal.cross_val_recall(clf)
# print "ExtraTreesClassifier : ", scores.mean()

# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400, algorithm="SAMME.R")

# scores = crossVal.cross_val_recall(clf)
# print "AdaBoostClassifier : ", scores.mean()

clf = linear_model.LogisticRegression()

scores = crossVal.cross_val_recall(clf)
print "LogisticRegression : ", scores.mean()

# clf = MultinomialNB()

# scores = crossVal.cross_val_recall(clf)
# print "MultinomialNB : ", scores.mean()

# clf = svm.SVC(kernel = 'linear', C = 0.01)

# scores = crossVal.cross_val_recall(clf)
# print "SVM with linear kernel (C = 0.01): ", scores.mean()

# clf = svm.SVC(kernel='rbf', gamma = 1, C = 50)

# scores = crossVal.cross_val_recall(clf)
# print "SVM gaussian kernel (C = 50, gamma = 1): ", scores.mean()

print ""
print "F1 Scores"
print ""

clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

scores = crossVal.cross_val_f1score(clf)

#print scores
print "RandomForestClassifier : ", scores.mean()

# clf = ExtraTreesClassifier(n_estimators=120, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))

# scores = crossVal.cross_val_f1score(clf)
# print "ExtraTreesClassifier : ", scores.mean()

# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400, algorithm="SAMME.R")

# scores = crossVal.cross_val_f1score(clf)
# print "AdaBoostClassifier : ", scores.mean()

clf = linear_model.LogisticRegression()

scores = crossVal.cross_val_f1score(clf)
print "LogisticRegression : ", scores.mean()

# clf = MultinomialNB()

# scores = crossVal.cross_val_f1score(clf)
# print "MultinomialNB : ", scores.mean()

# clf = svm.SVC(kernel = 'linear', C = 0.01)

# scores = crossVal.cross_val_f1score(clf)
# print "SVM with linear kernel (C = 0.01): ", scores.mean()

# clf = svm.SVC(kernel='rbf', gamma = 1, C = 50)

# scores = crossVal.cross_val_f1score(clf)
# print "SVM gaussian kernel (C = 50, gamma = 1): ", scores.mean()

print ""


