import numpy as np 
import sys

from pprint import pprint

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Activation, Merge, Reshape, Input, LSTM, Dense, Bidirectional, Dropout
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import skipgrams, make_sampling_table, pad_sequences
from keras.utils import np_utils
from keras.models import load_model

from gensim.models.word2vec import Word2Vec

MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
num_epochs = 30
batches_size = 128
nb_classes = 2

print("read file")
#####training label
print("training_label.txt")
texts = []

training_label = open(sys.argv[1], "r")
for r in training_label.readlines():
	i = r.find('+++$+++')
	t = r[i+8:-1]
	texts.append(t)
training_label.close()

#####training nolabel
print("training_nolabel.txt")
training_nolabel = open(sys.argv[2], "r")

for r in training_nolabel.readlines():
	t = r[:-1]
	texts.append(t)
training_nolabel.close()

"""
###testing data
print("testing_data.txt")
testing_data = open(sys.argv[4], "r")
idx = 0

for r in testing_data.readlines():
	if idx != 0:
		i = r.find(',')
		t = r[i+1:-1]
		texts.append(t)
	idx += 1
testing_data.close()
"""
# tokenize
tokenized_corpus = []
for i in range(len(texts)):
	tokenized_corpus.append(text_to_word_sequence(texts[i])+["<PAD>"])

#for i in range(len(tokenized_corpus)):
#	tokenized_corpus[i] = tokenized_corpus[i] + ["<PAD>"]*(MAX_SEQUENCE_LENGTH - len(tokenized_corpus[i]))


vector_size = 512
window_size = 5
print("building Word2Vec model")

word2vec = Word2Vec(sentences=tokenized_corpus, min_count=1, size=vector_size,  window=window_size, negative=20, iter=10, sg=1)

print("saving Word2Vec model")
word2vec.save('./word2vec.model')

print("loading Word2Vec model")
word2vec = Word2Vec.load('./word2vec.model')

W2V = word2vec.wv


del word2vec
del tokenized_corpus
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(texts)

#word_index = tokenizer.word_index
#print(word_index)
#voc_size = len(word_index)+1

#print("voc_size = %d"%voc_size)


#sequences = tokenizer.texts_to_sequences(texts[:training_label_size])

#data1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#define the model

#embedding_layer = Embedding(input_dim=voc_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)

#sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#embedded_sequences = embedding_layer(sequence_input)

#print(embedded_sequences)
"""
x = LSTM(64, activation='sigmoid')(embedded_sequences)

#x = Bidirectional(LSTM(64, activation='sigmoid')(embedded_sequences))

x = Dense(units=512,activation='relu')(x)

x = Dense(units=128,activation='relu')(x)

preds = Dense(2, activation='softmax')(x)

print(preds)

model = Model(sequence_input, preds)
"""

model = Sequential()

#model.add(Embedding(input_dim=vector_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.3, input_shape=(MAX_SEQUENCE_LENGTH, vector_size)))
model.add(LSTM(units=100, activation='relu'))
#model.add(Bidirectional(LSTM(units=100, activation='relu')))
model.add(Dropout(0.3))
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=2,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

#if sys.argv[3] == '--train_supervised':
print('Train')
	#model = load_model('model-W2V-LSTM8.h5')
texts1 = []
labels1 = []

training_label = open(sys.argv[1], "r")
for r in training_label.readlines():
	i = r.find('+++$+++')
	t = r[i+8:-1]
	texts1.append(t)
	labels1.append(int(r[0]))
training_label.close()

labels1 = np.array(labels1)
labels1 = np_utils.to_categorical(labels1, nb_classes)

data1 = []

for i in range(len(texts1)):
	data1.append(text_to_word_sequence(texts1[i]))

training_label_max_length = -1

for r in data1:
	if len(r) > training_label_max_length:
		training_label_max_length = len(r)

print(training_label_max_length)

for i in range(len(data1)):
	data1[i] = data1[i] + ["<PAD>"]*(MAX_SEQUENCE_LENGTH - len(data1[i]))

for i in range(len(data1)):
	for j in range(len(data1[i])):
		data1[i][j] = W2V[data1[i][j]]

data1 = np.array(data1)

print('\nlabel\n')
for idx_epoch in range(num_epochs):
	loss = 0
	acc = 0
	start = 0
	end = start + batches_size
	while start < data1.shape[0]:
		x = data1[start:end]
		y = labels1[start:end]
		score = model.train_on_batch(x, y)
		loss += score[0]
		acc += score[1]
		start += batches_size
		end = start + batches_size
	print(idx_epoch)
	print(loss)
	print(acc)
	model.save('model-W2V-LSTM%d.h5'%(idx_epoch+1))
"""
else:
	print("Testing\n")
	model = load_model('model-W2V-LSTM30.h5')
	testing_data = open(sys.argv[4], "r")

	texts3 = []
	test_id = []
	idx = 0
	for r in testing_data.readlines():
		if idx != 0:
			i = r.find(',')
			test_id.append(int(r[:i]))
			t = r[i+1:-1]
			texts3.append(t)
		idx += 1
	testing_data.close()

	data3 = []

	for i in range(len(texts3)):
		data3.append(text_to_word_sequence(texts3[i]))

	testing_data_max_length = -1

	for r in data3:
		if len(r) > testing_data_max_length:
			testing_data_max_length = len(r)

	print(testing_data_max_length)

	for i in range(len(data3)):
		data3[i] = data3[i] + ["<PAD>"]*(MAX_SEQUENCE_LENGTH - len(data3[i]))

	for i in range(len(data3)):
		for j in range(len(data3[i])):
			data3[i][j] = W2V[data3[i][j]]

	data3 = np.array(data3)
	#sequences3 = tokenizer.texts_to_sequences(texts3)

	#data3 = pad_sequences(sequences3, maxlen=MAX_SEQUENCE_LENGTH)
	test_label = model.predict(data3, batch_size=batches_size)
	print("test_label:")
	print(test_label)
	result_csv = open(sys.argv[5], "w")
	result_csv.write("id,label\n")
	for i in range(len(test_id)):
		result_csv.write(str(test_id[i])+","+str(np.argmax(test_label[i]))+"\n")

	result_csv.close()
"""