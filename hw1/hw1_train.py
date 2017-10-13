# -*- coding : utf-8 -*-
import numpy
import codecs
import sys
import csv
import pandas as pd 
import numpy as np
import pdb
#data = pd.read_csv(sys.argv[1])
#print(data)

train_data = codecs.open(sys.argv[1], "r", encoding='utf-8', errors='ignore')
test_data = codecs.open(sys.argv[2], "r", encoding='utf-8', errors='ignore')
output = open("./result/output.csv", "w")

train_dataset = []

reader1 = csv.reader(train_data, delimiter=",")
for line in reader1:
	t = line[2:]
	for i in range(1, len(t)):
		if t[i] == "NR":
			t[i] = float(0.0)
		else:
			t[i] = float(t[i])

	train_dataset.append(t)

test_dataset = []

reader2 = csv.reader(test_data, delimiter=",")
for line in reader2:
	for i in range(2, len(line)):
		if line[i] == "NR":
			line[i] = float(0.0)
		else:
			line[i] = float(line[i])

	test_dataset.append(line)


train_data_array = []
for i in train_dataset[1:]:
	train_data_array.append([i[0], np.array(i[1:])])


test_data_array = []
for i in test_dataset:
	test_data_array.append([i[0], i[1], np.array(i[2:])])



#concat nextday info to lastday info for train_data_array
for i in range(0, len(train_data_array)-18):
	train_data_array[i][1] = np.hstack((train_data_array[i][1], train_data_array[i+18][1]))
train_data_array[len(train_data_array)-18][1] = np.hstack((train_data_array[len(train_data_array)-18][1], train_data_array[0][1][0:24]))
train_data_array[len(train_data_array)-17][1] = np.hstack((train_data_array[len(train_data_array)-17][1], train_data_array[1][1][0:24]))
train_data_array[len(train_data_array)-16][1] = np.hstack((train_data_array[len(train_data_array)-16][1], train_data_array[2][1][0:24]))
train_data_array[len(train_data_array)-15][1] = np.hstack((train_data_array[len(train_data_array)-15][1], train_data_array[3][1][0:24]))
train_data_array[len(train_data_array)-14][1] = np.hstack((train_data_array[len(train_data_array)-14][1], train_data_array[4][1][0:24]))
train_data_array[len(train_data_array)-13][1] = np.hstack((train_data_array[len(train_data_array)-13][1], train_data_array[5][1][0:24]))
train_data_array[len(train_data_array)-12][1] = np.hstack((train_data_array[len(train_data_array)-12][1], train_data_array[6][1][0:24]))
train_data_array[len(train_data_array)-11][1] = np.hstack((train_data_array[len(train_data_array)-11][1], train_data_array[7][1][0:24]))
train_data_array[len(train_data_array)-10][1] = np.hstack((train_data_array[len(train_data_array)-10][1], train_data_array[8][1][0:24]))
train_data_array[len(train_data_array)-9][1] = np.hstack((train_data_array[len(train_data_array)-9][1], train_data_array[9][1][0:24]))
train_data_array[len(train_data_array)-8][1] = np.hstack((train_data_array[len(train_data_array)-8][1], train_data_array[10][1][0:24]))
train_data_array[len(train_data_array)-7][1] = np.hstack((train_data_array[len(train_data_array)-7][1], train_data_array[11][1][0:24]))
train_data_array[len(train_data_array)-6][1] = np.hstack((train_data_array[len(train_data_array)-6][1], train_data_array[12][1][0:24]))
train_data_array[len(train_data_array)-5][1] = np.hstack((train_data_array[len(train_data_array)-5][1], train_data_array[13][1][0:24]))
train_data_array[len(train_data_array)-4][1] = np.hstack((train_data_array[len(train_data_array)-4][1], train_data_array[14][1][0:24]))
train_data_array[len(train_data_array)-3][1] = np.hstack((train_data_array[len(train_data_array)-3][1], train_data_array[15][1][0:24]))
train_data_array[len(train_data_array)-2][1] = np.hstack((train_data_array[len(train_data_array)-2][1], train_data_array[16][1][0:24]))
train_data_array[len(train_data_array)-1][1] = np.hstack((train_data_array[len(train_data_array)-1][1], train_data_array[17][1][0:24]))




#split into training set and validation set

#for i in range(0, len(train_data_array), interval):
 

#step1: define function set
bias = float(0.0)
w = np.zeros(9)
y = []
for i in range(0, len(train_data_array), 18):
	#for j in range(0, 24):
	y_t = bias
		#idx = 0
		#while idx < 18:
	y_t += np.sum(w * train_data_array[i+9][1][:9])
		#idx += 1
	y.append(y_t)

#step2: define loss function
#print(len(data_array))
#print(y, len(y))
#input()
lamda = 0.0
"""
Loss = []
for i in range(0, len(y)):
	Loss.append((data_array[i*18+10][1][9] - y)**2 + lamda*np.sum(w*w))
"""
#step3: Gradient Descent
#pdb.set_trace()
numIterations = 500000
eata = 0.00000001
for i in range(0, numIterations):

	print(i)

	gradient_w = np.zeros(9)
	gradient_b = 0.0
	
	#for j in range(0, 18):  #which feature
	for k in range(0, 9):  #which hours
		t = 0.0
		for l in range(0, 240):   #which days
			#for m in range(0, 24): 
			t += 2 * (train_data_array[l*18+9][1][9] - y[l]) * (-train_data_array[l*18+9][1][k])
				
		t += lamda*2*w[k]
		gradient_w[k] = t

	t = 0.0
	for j in range(0, 240):
		#for k in range(0, 24):
		t += 2 * (train_data_array[j*18+9][1][9] - y[j])
	gradient_b = t

	w = w - eata * gradient_w	
	#print(w)
	#input()
	bias = bias - eata * gradient_b
	#print(bias)

	for j in range(0, len(train_data_array), 18):
		#for k in range(0, 24):
		y_t = bias
		#idx = 0
		#while idx < 18:
		y_t += np.sum(w * train_data_array[j+9][1][:9])
		#idx += 1
		y[(j//18)] = y_t
	#print(y, len(y))
	#input()

np.save('model_hw1.npy',[w, bias])

[w,bias] = np.load('model_hw1.npy')
#print(y, len(y))
#final step: output predicted value for the test data
#pdb.set_trace()
output.write("id,value\n")
for i in range(0, len(test_data_array), 18):
	y_t = bias
	#idx = 0
	#while idx < 18:
	y_t += np.sum(w * test_data_array[i+9][2][:9])
		#idx += 1
	output.write(test_data_array[i][0]+","+str(y_t)+"\n")

"""
for e in y:
	print(e)
	"""
"""
for i in dataset:
	print(i)
"""

"""
for line in data.readlines('\r\n'):
	t = line.split(",")
	s = t[2:]
	print(s)
"""

train_data.close()
test_data.close()
output.close()
