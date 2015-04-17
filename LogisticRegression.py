# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 17:02:36 2014

@author: Forrest
"""


import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

cnt=0
ary=np.genfromtxt('cs-training.csv', skip_header=1, delimiter=',')

imp=Imputer(missing_values='NaN', strategy='mean', axis=0)
lbl=ary[:, 1]
allAry=ary[:, 2:ary.shape[1]-1]

imp.fit(allAry)
allAry=imp.transform(allAry)
#allAry, lbl=shuffle(allAry, lbl)

trainAry=allAry[0:10000-1, :]
trainLbl=lbl[0:10000-1]
testAry=allAry[10000:15000-1, :]
testLbl=lbl[10000:15000-1]




scaler=preprocessing.StandardScaler().fit(allAry)
trainAry=scaler.transform(trainAry)
testAry=scaler.transform(testAry)

clf = LogisticRegression()


#svmClf=svm.SVC(C=1.0, kernel='linear')
print clf.fit(trainAry, trainLbl).score(trainAry, trainLbl)
print clf.fit(trainAry, trainLbl).score(testAry, testLbl)
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