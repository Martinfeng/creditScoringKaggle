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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import cPickle
import gzip

cnt=0
ary=np.genfromtxt('cs-training.csv', skip_header=1, delimiter=',')

imp=Imputer(missing_values='NaN', strategy='mean', axis=0)
lbl=ary[:, 1]
allAry=ary[:, 2:ary.shape[1]-1]

imp.fit(allAry)
allAry=imp.transform(allAry)
#allAry, lbl=shuffle(allAry, lbl)

numTrainSpls=70000;
numValSpls=30000
numTestSpls=50000;
numTotal=numTrainSpls+numValSpls+numTestSpls;
feaDim=allAry.shape[1];

trainAry=allAry[0:numTrainSpls, :]
trainLbl=lbl[0:numTrainSpls]

validAry=allAry[numTrainSpls:numTrainSpls+numValSpls, :]
validLbl=lbl[numTrainSpls:numTrainSpls+numValSpls]

testAry=allAry[numTrainSpls+numValSpls:numTotal, :]
testLbl=lbl[numTrainSpls+numValSpls:numTotal]

#scaler=preprocessing.StandardScaler().fit(allAry)
#trainAry=scaler.transform(trainAry)
#validAry=scaler.transform(validAry)
#testAry=scaler.transform(testAry)

trainAry=trainAry.astype(np.float32, copy=False)
trainLbl=trainLbl.astype(np.int64, copy=False)
testAry=testAry.astype(np.float32, copy=False)
testLbl=testLbl.astype(np.int64, copy=False)
validAry=testAry.astype(np.float32, copy=False)
validLbl=testLbl.astype(np.int64, copy=False)

train=(trainAry, trainLbl)
valid=(validAry, validLbl)
test=(testAry, testLbl)

sets=(train, valid, test)
output=gzip.open('creditdataunscaled.pkl.gz', 'wb')
cPickle.dump(sets, output, -1)
output.close() 
        
