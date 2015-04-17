#-*- coding: utf8 -*-
"""
@author: siyao

ExtraTreesClassifier
"""


import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import svm, cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

cnt=0
ary=np.genfromtxt('cs-training.csv', skip_header=1, delimiter=',')

imp=Imputer(missing_values='NaN', strategy='median', axis=0)
lbl=ary[:, 1]
allAry=ary[:, 2:ary.shape[1]-1]

imp.fit(allAry)
allAry=imp.transform(allAry)
#allAry, lbl=shuffle(allAry, lbl)

trainAry=allAry[0:20000, :]
trainLbl=lbl[0:20000]
testAry=allAry[20000:30000, :]
testLbl=lbl[20000:30000]

sampleWeight=np.empty([trainLbl.shape[0]], dtype=np.float32)
cnt=0
for curLbl in trainLbl:
    if curLbl==1:
        sampleWeight[cnt]=13.96
    if curLbl==0:
        sampleWeight[cnt]=1.0
    cnt=cnt+1
#scaler=preprocessing.StandardScaler().fit(allAry)
#trainAry=scaler.transform(trainAry)
#testAry=scaler.transform(testAry)
clf = ExtraTreesClassifier(n_estimators=5000, max_depth=None, min_samples_split=1, random_state=0)
#clf = RandomForestClassifier(n_estimators=6000, max_depth=7,  min_samples_split=1, random_state=0)



#svmClf=svm.SVC(C=1.0, kernel='linear')
#print clf.score(trainAry, trainLbl, scoring='roc_auc')
probs=clf.fit(trainAry, trainLbl, sampleWeight).predict_proba(testAry)

fpr, tpr, thresholds=roc_curve(testLbl, probs[:,1])

roc_auc = auc(fpr, tpr)
print roc_auc




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
