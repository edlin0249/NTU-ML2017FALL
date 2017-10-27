#import library
import numpy as np
import math
import sys
x_train_dms = open(sys.argv[1], "r")
y_train_dms = open(sys.argv[2], "r")

#arrange X_train.dms
x_train_dms_items = []

idx = 0
for r in x_train_dms.readlines():
	if idx != 0 and idx <= 6000:
		r = r[:len(r)-1]
		t = r.split(",")
		t = list(map(lambda x: float(x), t))
		x_train_dms_items.append(t)
	idx += 1

x_train_dms_items = np.array(x_train_dms_items)


#normalize
epsilon = 0.0001
mean_train_x = np.mean(x_train_dms_items, axis=0)
std_train_x = (np.sum((x_train_dms_items - mean_train_x) ** 2, axis=0)/len(x_train_dms_items)) ** 0.5
x_train_dms_items = (x_train_dms_items - mean_train_x)/(std_train_x+epsilon)

x_train_dms.close()

#arranfe Y_train.dms
y_train_dms_items = []

idx = 0
for r in y_train_dms.readlines():
	if idx != 0 and idx <= 6000:
		r = int(r[0])
		y_train_dms_items.append(r)
	idx += 1

y_train_dms.close()

x = np.array(x_train_dms_items)
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
y = np.array(y_train_dms_items)
W = np.zeros(len(x[0]))
l_rate = 0.5

repeat = 426
D = np.zeros(len(x[0]))
x_t = x.transpose()
G = np.zeros(len(x[0]))
#lamda = 0.01

for i in range(repeat):
	prob = 1/(1+np.exp(-np.dot(x, W)))
	loss = prob - y
	#cost = np.sum(loss**2) / len(x)
	#cost_a  = math.sqrt(cost)
	gra = np.dot(x_t,loss)#+lamda*2*W
	G = l_rate*G+(1-l_rate)*(gra**2)
	delta_w = -(np.sqrt(D)+epsilon)/(np.sqrt(G)+epsilon)*gra
	D = l_rate*D+(1-l_rate)*(delta_w**2)
	#s_gra += gra**2
	#ada = np.sqrt(s_gra)
	W = W + delta_w
	p = 1/(1+np.exp(-np.dot(x, W)))
	offset = 0.001
	Cross_entropy = -np.sum(y*np.log(p+offset)+(1-y)*np.log(1-p+offset))
	print ('iteration: %d, cross entropy: %f' % ( i, Cross_entropy))

np.save('model_logistic.npy', W)
np.save('model_logistic_mean.npy', mean_train_x)
np.save('model_logistic_std.npy', std_train_x)
W = np.load('model_logistic.npy')
mean_train_x = np.load('model_logistic_mean.npy')
std_train_x = np.load('model_logistic_std.npy')


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
