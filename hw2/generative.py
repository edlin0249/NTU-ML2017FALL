#import library
import numpy as np
from math import exp
import sys
x_train_dms = open(sys.argv[1], "r")
y_train_dms = open(sys.argv[2], "r")

#arrange X_train.dms
x_train_dms_items = []

idx = 0
for r in x_train_dms.readlines():
	if idx != 0:
		r = r[:len(r)-1]
		t = r.split(",")
		t = list(map(lambda x: float(x), t))
		x_train_dms_items.append(t)
	idx += 1

x_train_dms_items = np.array(x_train_dms_items)
print(x_train_dms_items.shape[0])
mean_train_x = np.mean(x_train_dms_items, axis=0)
std_train_x = np.std(x_train_dms_items, axis=0)
x_train_dms_items = (x_train_dms_items - mean_train_x)/std_train_x

x_train_dms.close()

#arranfe Y_train.dms
y_train_dms_items = []

idx = 0
for r in y_train_dms.readlines():
	if idx != 0:
		r = int(r[0])
		y_train_dms_items.append(r)
	idx += 1

y_train_dms.close()
#category
class1 = []
class2 = []

for i in range(len(y_train_dms_items)):
	if y_train_dms_items[i] == 1:
		class1.append(x_train_dms_items[i])
	else:
		class2.append(x_train_dms_items[i])

class1 = np.array(class1)
#print(class1)
class2 = np.array(class2)

#find mean and covariance matrix
class1 = class1.T
#print(class1.shape)
mean1 = np.mean(class1, axis=1)
#print(mean1.shape)
cov1 = np.cov(class1)
#print(cov1.shape)
class2 = class2.T
mean2 = np.mean(class2, axis=1)
cov2 = np.cov(class2)
#print(cov2.shape)
total = class1.shape[1]+class2.shape[1] #how many items
print(total)
cov = class1.shape[1]/total*cov1 + class2.shape[1]/total*cov2
#print(cov.shape)
try:
	W = np.dot((mean1-mean2).T, np.linalg.inv(cov))
	b = -1/2*np.dot(mean1.T, np.dot(np.linalg.inv(cov), mean1))+1/2*np.dot(mean2.T, np.dot(np.linalg.inv(cov), mean2))+np.log(class1.shape[1]/class2.shape[1])
except numpy .linalg.LinAlgError:
	W = np.dot((mean1-mean2).T, np.linalg.pinv(cov))
	b = -1/2*np.dot(mean1.T, np.dot(np.linalg.pinv(cov), mean1))+1/2*np.dot(mean2.T, np.dot(np.linalg.pinv(cov), mean2))+np.log(class1.shape[1]/class2.shape[1])

np.save("model_generative_b.npy", b)
np.save("model_generative_W.npy", W)
np.save("model_generative_mean.npy", mean_train_x)
np.save("model_generative_std.npy", std_train_x)
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
