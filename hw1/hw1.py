# -*- coding : utf-8 -*-
import numpy
import codecs
import sys
import csv
import pandas as pd 
import numpy as np
import pdb

test_data = codecs.open(sys.argv[1], "r", encoding='utf-8', errors='ignore')
output = open(sys.argv[2], "w")

test_dataset = []

reader2 = csv.reader(test_data, delimiter=",")
for line in reader2:
	for i in range(2, len(line)):
		if line[i] == "NR":
			line[i] = float(0.0)
		else:
			line[i] = float(line[i])

	test_dataset.append(line)

test_data_array = []
for i in test_dataset:
	test_data_array.append([i[0], i[1], np.array(i[2:])])

w = np.zeros(9)
bias = float(0.0)

[w,bias] = np.load('model_hw1.npy')

output.write("id,value\n")
for i in range(0, len(test_data_array), 18):
	y_t = bias
	#idx = 0
	#while idx < 18:
	y_t += np.sum(w * test_data_array[i+9][2][:9])
		#idx += 1
	output.write(test_data_array[i][0]+","+str(y_t)+"\n")

test_data.close()
output.close()
