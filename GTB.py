#-*- coding: utf8 -*-
"""
@author: siyao

GradientBoostingClassifier
"""

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
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

trainAry=allAry[0:100000, :]
trainLbl=lbl[0:100000]
testAry=allAry[100000:150000, :]
testLbl=lbl[100000:150000]

scaler=preprocessing.StandardScaler().fit(allAry)
trainAry=scaler.transform(trainAry)
testAry=scaler.transform(testAry)

clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)


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
