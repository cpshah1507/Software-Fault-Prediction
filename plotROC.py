import numpy as np
from roc import plotroc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

allData = np.loadtxt("JM1.arff", delimiter=',')

X = allData[:,:-1]
y = allData[:,-1]
n_features = X.shape[1]

lrmodel = LogisticRegression()
plotroc(X,y,N=500,k=5,clf=lrmodel,modelname="Logistic Regression")

rfmodel = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))
plotroc(X,y,N=500,k=5,clf=rfmodel,modelname="Random Forests")

etmodel = ExtraTreesClassifier(n_estimators=120, max_depth=None, min_samples_split=1, random_state=0, max_features=int(pow(n_features, 0.5)))
plotroc(X,y,N=500,k=5,clf=etmodel,modelname="Extra Trees")

abmodel = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400, algorithm="SAMME.R")
plotroc(X,y,N=500,k=5,clf=abmodel,modelname="Ada Boost")

