from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import SVC
#from sklearn.metrics.pairwise import cosine_similarity

from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


dataFiles = ["PC4.arff"]
fpr1 = dict()
tpr1 = dict()
roc_auc1 = dict()

for fname in dataFiles:
	# load the data
	allData = np.loadtxt(fname, delimiter=',')


	X = allData[:,:-1]
	y = allData[:,-1]
	
	plt.figure()
	
	kf = cross_validation.KFold(len(X), n_folds=10) 
	for train_index,test_index in kf:
		Xtrain = X[train_index]
		ytrain = y[train_index]
		Xtest = X[test_index]
		ytest = y[test_index]

		ytest = [int(el) for el in ytest]
		ytest1 = label_binarize(ytest,classes=[0,1,2])
		ytest1 = ytest1[:,:-1]
				
		clf = MultinomialNB()
		clf.fit(Xtrain,ytrain)

		ypred = clf.predict_proba(Xtest)

		fpr1["micro"], tpr1["micro"], _ = roc_curve(ytest1.ravel(), ypred.ravel())
		roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

		# Plot ROC curve
		plt.plot(fpr1["micro"], tpr1["micro"],label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc1["micro"]))
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic to multi-class')
		plt.legend(loc="lower right")

	plt.show()

