
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
        self.accuracy = {}
        self.precision = {}
        self.recall = {}
        self.f1score = {}

    def splitDataLabelWise(self, allData):
        temp = allData[allData[:,-1].argsort()]
        pos = np.where(temp[:, -1] == 1)[0][0]
        self.dataNeg = temp[: pos, :]
        self.dataPos = temp[pos:, :]
         
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

        self.listPos = indexPos
        self.listNeg = indexNeg
        return

    def cross_val_score(self, model):

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        for i in range(self.folds):
            Xtrain, Ytrain, Xtest, Ytest = self.train_test_set(i)
            model.fit(Xtrain, Ytrain)
            Ypred = model.predict(Xtest)
            accuracy = accuracy_score(Ytest, Ypred)
            precision = precision_score(Ytest, Ypred, average='macro')
            recall = recall_score(Ytest, Ypred, average='macro')
            f1score = f1_score(Ytest, Ypred, average='macro')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1score)
        self.accuracy[model] = np.array(accuracy_scores)
        self.precision[model] = np.array(precision_scores)
        self.recall[model] = np.array(recall_scores)
        self.f1score[model] = np.array(f1_scores)
        return 

    def cross_val_accuracy(self, model):
        if model not in self.accuracy:
            self.cross_val_score(model)
        return self.accuracy[model]

    def cross_val_precision(self, model):
        if model not in self.precision:
            self.cross_val_score(model)
        return self.precision[model]

    def cross_val_recall(self, model):
        if model not in self.recall:
            self.cross_val_score(model)    
        return self.recall[model]

    def cross_val_f1score(self, model):
        if model not in self.f1score:
            self.cross_val_score(model)
        return self.f1score[model]

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
