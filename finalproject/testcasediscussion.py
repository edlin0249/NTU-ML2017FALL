import sys
import pandas as pd 
import json
import numpy as np

with open(sys.argv[1], encoding='utf-8') as json_data:
	data = json.load(json_data)['data']

contexts, questions, ids = [], [], []
counter = 0

for d in data:
	paragraphs = d['paragraphs']
	for paragraph in paragraphs:
		context = paragraph['context']
		qas = paragraph['qas']
		for qa in qas:
			question = qa['question']
			qa_id = qa['id']

			contexts.append(context)
			questions.append(question)
			ids.append(qa_id)

			counter+=1

contexts = np.array(contexts)
questions = np.array(questions)
ids = np.array(ids)

res = open(sys.argv[2], 'r')
tmp = []
idx = 0
for i in res.readlines():
	if idx != 0:
		i = i.split(',')
		i[1] = i[1].split(' ')
		i[1] = list(map(lambda x:int(x), i[1]))
		tmp.append(i)
	idx += 1

#print(tmp[0])
#print(tmp[0][1])
#print(tmp[1])
#print(tmp[1][1])

wrfile = open(sys.argv[3], 'w')
wrfile.write('id,answer\n')
for i in range(counter):
	#print(tmp[i][1])
	t = []
	for j in tmp[i][1]:
		t.append(contexts[i][j])
	#t = contexts[i][tmp[i][1][0]:tmp[i][1][-1]]
	#print(t)
	wrfile.write(str(tmp[i][0])+','+"".join(t)+'\n')
wrfile.close()
