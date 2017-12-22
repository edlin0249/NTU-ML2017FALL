import sys
import pandas as pd 
import numpy as np
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Flatten, Dropout, Input, Dot, Add, Concatenate, Dense
from keras.layers.merge import dot, add
from keras.models import Sequential, Model
import os
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.02
set_session(tf.Session(config=config))
 
n_epochs = 30
latent_dim = 128
MODEL_WEIGHTS_FILE = 'ml1m_weights_train.h5'
l2regularizer=1e-4

ratings = pd.read_csv(sys.argv[1], sep=',', encoding='latin-1', usecols=['TrainDataID', 'UserID', 'MovieID', 'Rating'])

data = []
for i in range(len(ratings['TrainDataID'])):
	data.append([ratings['TrainDataID'][i], ratings['UserID'][i], ratings['MovieID'][i], ratings['Rating'][i]])

data = np.array(data)

np.random.seed(1019)
index = np.random.permutation(len(data))
data = data[index]

ratings['TrainDataID'] = data[:, 0]
ratings['UserID'] = data[:, 1]
ratings['MovieID'] = data[:, 2]
ratings['Rating'] = data[:, 3]

ratings['Rating'] = np.reshape(ratings['Rating'], (-1, 1))

n_users = ratings['UserID'].drop_duplicates().max()
n_movies = ratings['MovieID'].drop_duplicates().max()

ratings['UserID'] = ratings['UserID']-1

ratings['MovieID'] = ratings['MovieID']-1




mean = np.mean(ratings['Rating'])
std = np.std(ratings['Rating'])
#print("ratings['Rating']:")
#print(ratings['Rating'])
#ratings['Rating'] = (ratings['Rating'] - mean)/std
#print(ratings['Rating'])

user_input = Input(shape=[1])
movie_input = Input(shape=[1])
user_vec = Embedding(n_users, latent_dim)(user_input)
user_vec = Flatten()(user_vec)
user_vec = Dropout(0.3)(user_vec)
movie_vec = Embedding(n_movies, latent_dim)(movie_input)
movie_vec = Flatten()(movie_vec)
movie_vec = Dropout(0.3)(movie_vec)
"""
merge_vec = Concatenate()([user_vec, movie_vec])

hidden = Dense(150, activation='relu')(merge_vec)
hidden = Dropout(0.3)(hidden)
hidden = Dense(50, activation='relu')(hidden)
hidden = Dropout(0.3)(hidden)
r_hat = Dense(1)(hidden)
"""

user_bias = Embedding(n_users, 1)(user_input)
user_bias = Flatten()(user_bias)
movie_bias = Embedding(n_movies, 1)(movie_input)
movie_bias = Flatten()(movie_bias)

r_hat = Dot(axes=1)([user_vec, movie_vec])
r_hat = Add()([r_hat, user_bias, movie_bias])


#model = load_model(MODEL_WEIGHTS_FILE)
model = Model([user_input, movie_input], r_hat)

model.summary()
def rmse(y_true, y_pred): return K.sqrt( K.mean(((y_pred - y_true))**2) )
model.compile(optimizer='adam', loss='mse', metrics=[rmse])

es = EarlyStopping(monitor='val_rmse', patience=5, verbose=1, mode='min')
cp = ModelCheckpoint(monitor='val_rmse', save_best_only=True, save_weights_only=False, mode='min', filepath=MODEL_WEIGHTS_FILE)

history = model.fit([ratings['UserID'], ratings['MovieID']], ratings['Rating'], batch_size=10000, epochs=1000, validation_split=0.05, verbose=1, callbacks=[es, cp])

H = history.history
best_val = str( round(np.min(H['val_rmse']), 6) )
print('Best Val RMSE:', best_val)

#model.save('hw5_model_MF.h5')

testing_predict = pd.read_csv(sys.argv[2], sep=',', encoding='latin-1', usecols=['TestDataID','UserID','MovieID'])

testing_predict['UserID'] = testing_predict['UserID'] - 1

testing_predict['MovieID'] = testing_predict['MovieID'] - 1

prediction = model.predict([testing_predict['UserID'], testing_predict['MovieID']])
#prediction = prediction*std + mean

prediction = np.clip(prediction, 1, 5)
prediction = np.reshape(prediction, (-1,1))

print(prediction)
result_csv = open(sys.argv[3], 'w')
result_csv.write('TestDataID,Rating\n')
for i in range(len(testing_predict['TestDataID'])):
	result_csv.write(str(testing_predict['TestDataID'][i])+','+str(prediction[i][0])+'\n')
result_csv.close()
