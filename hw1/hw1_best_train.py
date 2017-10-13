#import library
import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pdb


#read data
data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open(sys.argv[1], 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

#parse data to (x,y)
x = []
y = []
# 每 12 個月
for i in range(12):
#if i != 6:
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s])
                x[471*i+j].append(data[t][480*i+j+s]**2)
        y.append(data[9][480*i+j+9])
"""
for i in range(8, 12):
#if i != 6:
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*(i-2)+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
"""
x = np.array(x)
y = np.array(y)

# add square term
# x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
"""
#feature scaling
mean = np.mean(x, axis=0)
std_dev = np.std(x, axis=0)
print(x)
print(mean)
print(std_dev)
pdb.set_trace()

x[:,1:] = (x[:,1:]-mean[1:])/std_dev[1:]
print(x)
print(mean)
print(std_dev)
pdb.set_trace()
"""

#init weight & other hyperparams
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 1000000
#lamda = 0.1

#check your ans with close form solution
# use close form to check whether ur gradient descent is good
# however, this cannot be used in hw1.sh 
# w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)

#start training
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = 2*np.dot(x_t,loss) #+ lamda*2*w
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

#save/read model
# save model
np.save('model_hw1_best.npy',w)
# read model
w = np.load('model_hw1_best.npy')

#read testing data 
test_x = []
n_row = 0
text = open(sys.argv[2] ,"r")
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

filename = "result/predict_all_feature_degtwo_3.csv"
text = open(filename, "w")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()