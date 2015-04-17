#-*- coding: utf8 -*-
"""
@author: siyao

RBF_SVM
"""


import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
 
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

clf = RandomForestClassifier(n_estimators=10, max_depth=None,  min_samples_split=2, random_state=0)


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
