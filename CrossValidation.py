
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

'''
    This class normalizes the data, ensures same proportion of labels in test set and train set
    and calculates accuracy score performing n-fold cross validation.

    Main function: cross_val_score(self, model, X, Y, cv) 

    PLEASE SEE testCrossVal.py as example on how to use this class. 
    Report any bugs at shallav.varma@gmail.com. :P
'''

class CrossValidation:

    def __init__(self, X, Y, cv = 10):
        self.folds = cv
        X = self.normalize(X)
        allData = np.c_[X, Y]
        self.folds = cv
        ratio = 1.0 / cv
        self.generateValTrainSets(allData, ratio)

    def splitDataLabelWise(self, allData):
        temp = allData[allData[:,-1].argsort()]
        pos = np.where(temp[:, -1] == 1)[0][0]
        self.dataNeg = allData[: pos, :]
        self.dataPos = allData[pos:, :]
         
    def shuffle(self, X):
        n, d = X.shape
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        return X

    def normalize(self, X):
        minX = np.amin(X, axis = 0)
        maxX = np.amax(X, axis = 0)
        den = maxX - minX
        X = (X - minX)/ den
        return X

    def instancesPerLabel(self):
        return self.dataPos.shape[0], self.dataNeg.shape[0]

    def generateValTrainSets(self, allData, ratio):
        self.splitDataLabelWise(allData)

        self.dataNeg = self.shuffle(self.dataNeg)
        self.dataPos = self.shuffle(self.dataPos)
        numNeg = int(self.dataNeg.shape[0] * ratio)
        numPos = int(self.dataPos.shape[0] * ratio)
        indexNeg = []
        indexPos = []
        firstNeg = 0
        firstPos = 0 
        for i in range(self.folds):
            lastNeg = firstNeg + numNeg
            lastPos = firstPos + numPos
            if i == self.folds - 1:
                indexNeg.append((firstNeg, self.dataNeg.shape[0] + 1))
                indexPos.append((firstPos, self.dataPos.shape[0] + 1))
            else:
                indexNeg.append((firstNeg, lastNeg))
                indexPos.append((firstPos, lastPos))
            firstPos = firstPos + numPos
            firstNeg = firstNeg + numNeg
        #print indexPos, indexNeg
        self.listPos = indexPos
        self.listNeg = indexNeg
        return

    def cross_val_accuracy(self, model):

        scores = []
        for i in range(self.folds):
            Xtrain, Ytrain, Xtest, Ytest = self.train_test_set(i)
            model.fit(Xtrain, Ytrain)
            Ypred = model.predict(Xtest)
            accuracy = accuracy_score(Ytest, Ypred)
            scores.append(accuracy)
        return np.array(scores)

    def cross_val_precision(self, model):

        precision_scores = []
        for i in range(self.folds):
            Xtrain, Ytrain, Xtest, Ytest = self.train_test_set(i)
            model.fit(Xtrain, Ytrain)
            Ypred = model.predict(Xtest)
            accuracy = precision_score(Ytest, Ypred, average='macro')
            precision_scores.append(accuracy)
        return np.array(precision_scores)

    def cross_val_recall(self, model):

        recall_scores = []
        for i in range(self.folds):
            Xtrain, Ytrain, Xtest, Ytest = self.train_test_set(i)
            model.fit(Xtrain, Ytrain)
            Ypred = model.predict(Xtest)
            accuracy = recall_score(Ytest, Ypred, average='macro')
            recall_scores.append(accuracy)
        return np.array(recall_scores)

    def cross_val_f1score(self, model):

        f1_scores = []
        for i in range(self.folds):
            Xtrain, Ytrain, Xtest, Ytest = self.train_test_set(i)
            model.fit(Xtrain, Ytrain)
            Ypred = model.predict(Xtest)
            accuracy = f1_score(Ytest, Ypred, average='macro')
            f1_scores.append(accuracy)
        return np.array(f1_scores)

    def train_test_set(self, i):
        listPos = self.listPos
        listNeg = self.listNeg
        dataNegVal = self.dataNeg[listNeg[i][0]:listNeg[i][1], :]
        dataNegTrain = np.append(self.dataNeg[:listNeg[i][0], :], self.dataNeg[listNeg[i][1]:, :], axis = 0)

        dataPosVal = self.dataPos[listPos[i][0]:listPos[i][1], :]
        dataPosTrain = np.append(self.dataPos[:listPos[i][0], :], self.dataPos[listPos[i][1]:, :], axis = 0)

        valSet = np.append(dataPosVal, dataNegVal, axis = 0)
        trainSet = np.append(dataPosTrain, dataNegTrain, axis = 0)

        valSet = self.shuffle(valSet)
        trainSet = self.shuffle(trainSet)

        Xtrain = trainSet[:, :-1]
        Ytrain = trainSet[:, -1]
        Xtest = valSet[:, :-1]
        Ytest = valSet[:, -1] 
        return Xtrain, Ytrain, Xtest, Ytest









