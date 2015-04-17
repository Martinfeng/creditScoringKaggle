#-*- coding: utf8 -*-
"""
@author: siyao
industries finance data interface, response is JSON style


Convert CSV file to libsvm format. Works only with numeric variables.
Put -1 as label index (argv[3]) if there are no labels in your file.
Expecting no headers. If present, headers can be skipped with argv[4] == 1.
"""

import sys
import csv

def construct_line( label, line, skip_first_col ):
	new_line = []
	if float( label ) == 0.0:
		label = "0"
	new_line.append( label )
	
	for i, item in enumerate( line ):
         if i==0 and 1==skip_first_col:
               continue
         if item == ''  or (item.isdigit() and float( item ) == 0.0):
			continue
         new_item = "%s:%s" % ( i, item )
         new_line.append( new_item )
	new_line = " ".join( new_line )
	new_line += "\n"
	return new_line

# ---
import numpy as np
from sklearn.preprocessing import Imputer



cnt=0
#ary=np.genfromtxt('train_v2.csv', skip_header=1, delimiter=',')
#imp=Imputer(missing_values='NaN', strategy='median', axis=0)

#imp.fit(ary)
#ary=imp.transform(ary)
#np.savetxt("train_v2pre.cvs", ary, delimiter=",")

input_file = 'cs-trainingpre.csv'
output_file = 'cs-train.txt'


label_index = 1
skip_headers = 0
skip_first_col = 1
 
i = open( input_file )
o = open( output_file, 'wb' )

reader = csv.reader( i )
if skip_headers:
	headers = reader.next()

for line in reader:
    if label_index == -1:
        label = 1
    else:
        label = line.pop( label_index )

    if float(label)>1:
        label = "1"
        
    new_line = construct_line( label, line, skip_first_col )
    o.write( new_line )
