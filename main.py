#-*- coding: utf8 -*-
"""
@author: siyao

"""


import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

cnt=0
ary=np.genfromtxt('cs-training.csv', skip_header=1, delimiter=',')
imp=Imputer(missing_values='NaN', strategy='mean', axis=0)
lbl=ary[:, 1]
allAry=ary[:, 2:ary.shape[1]-1]
imp.fit(allAry)
trainAry=ary[0:10000-1, 2:ary.shape[1]-1]
trainLbl=lbl[0:10000-1]
testAry=ary[10000:15000-1, 2:ary.shape[1]-1]
testLbl=lbl[10000:15000-1]

trainAry=imp.transform(trainAry)
testAry=imp.transform(testAry)
allAry=imp.transform(allAry)

scaler=preprocessing.StandardScaler().fit(allAry)
trainAry=scaler.transform(trainAry)
testAry=scaler.transform(testAry)

clf = svm.SVC(C=1, kernel='linear')
clf.fit(trainAry, trainLbl).score(trainAry, trainLbl)

C_range=10.0**np.arange(-2,2)
gamma_range=np.logspace(-6, 3, 18)
param_grid=dict(gamma=gamma_range, C=C_range)
cv=StratifiedKFold(y=trainLbl, n_folds=3)
grid=GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(trainAry, trainLbl)
bestE=grid.best_estimator_
print("The best classifier is:", grid.best_estimator_)


#svmClf=svm.SVC(C=1.0, kernel='linear')
print bestE.fit(trainAry, trainLbl).score(trainAry, trainLbl)
print bestE.fit(trainAry, trainLbl).score(testAry, testLbl)
#with open('C:\\CSSTEV\\LoanDefault\\train_v2.csv\\train_v2.csv', 'rb') as csvfile:
   # spamreader=csv.reader(csvfile, delimiter=',')
#    for x in csvfile.readlines():
#        splitObj=x.split(',')
#        ary=np.append(ary, splitObj)
#        cnt=cnt+1
    #for row in spamreader:
    #    if cnt>0:
    #        ary=np.append(ary, row)
    #    else:
    #        numCols=len(row)
    #    cnt=cnt+1
        
#numRows=cnt
#np.reshape(ary, [numRows, numCols])
