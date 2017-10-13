#import library
import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pdb

# read model
w = np.load('model_hw1_best.npy')

#read testing data 
test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
            test_x[n_row//18].append(float(r[i]) ** 2)
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
                test_x[n_row//18].append(float(r[i]) ** 2)
            else:
                test_x[n_row//18].append(0.0)
                test_x[n_row//18].append(0.0**2)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
#test_x[:,1:] = (test_x[:,1:] - mean[1:])/std_dev[1:]


#get ans.csv with your model 
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()