#import library
import numpy as np
from math import exp
import sys

b = np.load("model_generative_b.npy")
W = np.load("model_generative_W.npy")
mean_train_x = np.load("model_generative_mean.npy")
std_train_x = np.load("model_generative_std.npy")

#find probability
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

x_test_items = (x_test_items-mean_train_x)/std_train_x

y_test_csv = open(sys.argv[4], "w")
y_test_csv.write("id,label\n")
idx = 1
for t in x_test_items:
	prob = 1/(1+exp(-(np.dot(W, t)+b)))
	if prob >= 0.5:
		y_test_csv.write(str(idx)+",1\n")
	else:
		y_test_csv.write(str(idx)+",0\n")
	idx += 1
x_test_dms.close()
y_test_csv.close()
