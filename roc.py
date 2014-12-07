'''
If you run this file directly, it will run ROC analysis on the hardcoded data file given below.
Otherwise this file can be used by using exported function: plotroc
'''

from sklearn.naive_bayes import MultinomialNB
from CrossValidation import CrossValidation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from Smote import sampleData

def plotroc(X,y,N=200,k=5,clf=None,modelname="None"):
	if clf == None:
		print "Classifier not defined"
		return
	fpr1 = dict()
	tpr1 = dict()
	roc_auc1 = dict()
	folds = 2

	plt.figure()
	cv = CrossValidation(X,y,cv=folds)

	for i in xrange(folds):
		Xtrain, ytrain, Xtest, ytest = cv.train_test_set(i)
		ytest = [int(el) for el in ytest]
		ytest1 = label_binarize(ytest,classes=[0,1,2])
		ytest1 = ytest1[:,:-1]

		clf1 = OneVsRestClassifier(clf)
		ytrain1 = label_binarize(ytrain,classes=[0,1,2])
		ytrain1 = ytrain1[:,:-1]
		clf1.fit(Xtrain,ytrain1)
		ypred1 = clf1.predict_proba(Xtest)
		fpr1["model"], tpr1["model"], _ = roc_curve(ytest1.ravel(), ypred1.ravel())
		roc_auc1["model"] = auc(fpr1["model"], tpr1["model"])
		
	plt.plot(fpr1["model"], tpr1["model"],label='average ROC curve without Smote (area = {0:0.2f})'.format(roc_auc1["model"]))

	X, y = sampleData(X, y, N, 10, k)

	fpr1 = dict()
	tpr1 = dict()
	roc_auc1 = dict()
	folds = 2

	cv = CrossValidation(X,y,cv=folds)

	for i in xrange(folds):
		Xtrain, ytrain, Xtest, ytest = cv.train_test_set(i)
		ytest = [int(el) for el in ytest]
		ytest1 = label_binarize(ytest,classes=[0,1,2])
		ytest1 = ytest1[:,:-1]

		clf1 = OneVsRestClassifier(clf)
		ytrain1 = label_binarize(ytrain,classes=[0,1,2])
		ytrain1 = ytrain1[:,:-1]
		clf1.fit(Xtrain,ytrain1)
		ypred1 = clf1.predict_proba(Xtest)
		fpr1["model"], tpr1["model"], _ = roc_curve(ytest1.ravel(), ypred1.ravel())
		roc_auc1["model"] = auc(fpr1["model"], tpr1["model"])
		
	plt.plot(fpr1["model"], tpr1["model"],label='average ROC curve with Smote (area = {0:0.2f})'.format(roc_auc1["model"]))

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic for ' + modelname)
	plt.legend(loc="lower right")	
	plt.show()


if __name__ == "__main__":

	dataFiles = ["filtered.arff"]
	fpr1 = dict()
	tpr1 = dict()
	roc_auc1 = dict()
	folds = 2

	for fname in dataFiles:
		# load the data
		allData = np.loadtxt(fname, delimiter=',')

		X = allData[:,:-1]
		y = allData[:,-1]
		#X, y = sampleData(X, y, 200, 10, 5)
		n_features = X.shape[1]

		plt.figure()
		cv = CrossValidation(X,y,cv=folds)

		for i in xrange(folds):
			Xtrain, ytrain, Xtest, ytest = cv.train_test_set(i)

			ytest = [int(el) for el in ytest]
			ytest1 = label_binarize(ytest,classes=[0,1,2])
			ytest1 = ytest1[:,:-1]
			
			clf = MultinomialNB()
			clf.fit(Xtrain,ytrain)

			ypred = clf.predict_proba(Xtest)

			fpr1["nb"], tpr1["nb"], _ = roc_curve(ytest1.ravel(), ypred.ravel())
			roc_auc1["nb"] = auc(fpr1["nb"], tpr1["nb"])

			# Logistic Regression
			lrmodel = LogisticRegression()
			clf = OneVsRestClassifier(lrmodel)
			ytrain1 = label_binarize(ytrain,classes=[0,1,2])
			ytrain1 = ytrain1[:,:-1]
			clf.fit(Xtrain,ytrain1)
			ypred1 = clf.predict_proba(Xtest)
			
			fpr1["lr"], tpr1["lr"], _ = roc_curve(ytest1.ravel(), ypred1.ravel())
			roc_auc1["lr"] = auc(fpr1["lr"], tpr1["lr"])

			# Random Forest
			rfmodel = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))
			clf = OneVsRestClassifier(rfmodel)
			ytrain1 = label_binarize(ytrain,classes=[0,1,2])
			ytrain1 = ytrain1[:,:-1]
			clf.fit(Xtrain,ytrain1)
			ypred1 = clf.predict_proba(Xtest)
			
			fpr1["rf"], tpr1["rf"], _ = roc_curve(ytest1.ravel(), ypred1.ravel())
			roc_auc1["rf"] = auc(fpr1["rf"], tpr1["rf"])

			# Extra Trees
			etmodel = ExtraTreesClassifier(n_estimators=120, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))
			clf = OneVsRestClassifier(etmodel)
			ytrain1 = label_binarize(ytrain,classes=[0,1,2])
			ytrain1 = ytrain1[:,:-1]
			clf.fit(Xtrain,ytrain1)
			ypred1 = clf.predict_proba(Xtest)
			
			fpr1["et"], tpr1["et"], _ = roc_curve(ytest1.ravel(), ypred1.ravel())
			roc_auc1["et"] = auc(fpr1["et"], tpr1["et"])

			# Ada Boost
			abmodel = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400, algorithm="SAMME.R")
			clf = OneVsRestClassifier(abmodel)
			ytrain1 = label_binarize(ytrain,classes=[0,1,2])
			ytrain1 = ytrain1[:,:-1]
			clf.fit(Xtrain,ytrain1)
			ypred1 = clf.predict_proba(Xtest)
			
			fpr1["ab"], tpr1["ab"], _ = roc_curve(ytest1.ravel(), ypred1.ravel())
			roc_auc1["ab"] = auc(fpr1["ab"], tpr1["ab"])

			
		plt.plot(fpr1["nb"], tpr1["nb"],label='average ROC curve Naive Bayes (area = {0:0.2f})'.format(roc_auc1["nb"]))
		plt.plot(fpr1["lr"], tpr1["lr"],label='average ROC curve Logistic Regression (area = {0:0.2f})'.format(roc_auc1["lr"]))
		plt.plot(fpr1["rf"], tpr1["rf"],label='average ROC curve Random Forests (area = {0:0.2f})'.format(roc_auc1["rf"]))
		plt.plot(fpr1["et"], tpr1["et"],label='average ROC curve Extra Trees (area = {0:0.2f})'.format(roc_auc1["et"]))
		plt.plot(fpr1["ab"], tpr1["ab"],label='average ROC curve AdaBoost (area = {0:0.2f})'.format(roc_auc1["ab"]))


		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic to multi-class')
		plt.legend(loc="lower right")	
		plt.show()
