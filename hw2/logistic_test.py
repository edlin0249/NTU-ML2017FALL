#import library
import numpy as np
import math
import sys

W = np.load('model_logistic.npy')
mean_train_x = np.load('model_logistic_mean.npy')
std_train_x = np.load('model_logistic_std.npy')
epsilon = 0.0001
x_test_dms = open(sys.argv[3], "r")
x_test_items = []
idx = 0
for r in x_test_dms.readlines():
	if idx != 0:
		r = r[:len(r)-1]
		t = r.split(",")
		t = list(map(lambda x: float(x), t))
		x_test_items.append(t)
	idx += 1

x_test_items = (x_test_items-mean_train_x)/(std_train_x+epsilon)
b = W[0]
W = W[1:]

y_test_csv = open(sys.argv[4], "w")
y_test_csv.write("id,label\n")
idx = 1
for t in x_test_items:
	t = np.exp(-(np.dot(W, t)+b))
	#print(t)
	prob = 1/(1+t)
	if prob >= 0.5:
		y_test_csv.write(str(idx)+",1\n")
	else:
		y_test_csv.write(str(idx)+",0\n")
	idx += 1
x_test_dms.close()
y_test_csv.close()
