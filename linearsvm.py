#-*- coding: utf8 -*-
"""
@author: siyao

Linear SVM
"""


import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import svm, cross_validation
from sklearn.metrics import roc_curve, auc

cnt=0
ary=np.genfromtxt('cs-training.csv', skip_header=1, delimiter=',')
imp=Imputer(missing_values='NaN', strategy='median', axis=0)
lbl=ary[:, 1]
allAry=ary[:, 2:ary.shape[1]-1]
imp.fit(allAry)
trainAry=ary[0:100000, 2:ary.shape[1]-1]
trainLbl=lbl[0:100000]
testAry=ary[100000:150000, 2:ary.shape[1]-1]
testLbl=lbl[100000:150000]

trainAry=imp.transform(trainAry)
testAry=imp.transform(testAry)
allAry=imp.transform(allAry)

scaler=preprocessing.StandardScaler().fit(allAry)
trainAry=scaler.transform(trainAry)
testAry=scaler.transform(testAry)

clf = svm.SVC(C=10, kernel='linear', probability=True, class_weight={1: 14})

probs=clf.fit(trainAry, trainLbl).predict_proba(testAry)

fpr, tpr, thresholds=roc_curve(testLbl, probs[:,1])

roc_auc = auc(fpr, tpr)
print roc_auc

#print clf.score(trainAry, trainLbl)
#print clf.score(testAry, testLbl)
#print cross_validation.cross_val_score(clf, allAry, lbl, scoring='roc_auc', cv=2)
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
