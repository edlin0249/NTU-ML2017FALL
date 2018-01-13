import sys
import numpy as np 
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.02
set_session(tf.Session(config=config))

from keras.models import Model, load_model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Flatten, Reshape
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
import keras.backend as K

from sklearn.cluster import KMeans

import pickle


AUTOENCODER_FILE = 'hw6_model.h5'
KMEANS_FILE = 'kmeans.sav'
#ENCODER_FILE = 'encoder.h5'
images = np.load(sys.argv[1])

images = images.astype('float32') / 255.
images = np.reshape(images, (len(images), np.prod(images.shape[1:])))  # adapt this if using `channels_first` image data format


#while(1):
#autoencoder.fit(images, images, epochs=5, batch_size=256, shuffle=True, verbose=1)
#autoencoder.save(AUTOENCODER_FILE)
#encoder.save(ENCODER_FILE)
#del autoencoder
#del encoder
autoencoder = load_model(AUTOENCODER_FILE)
#autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, verbose=1, validation_data=[x_test, x_test], callbacks=[es, cp, TensorBoard(log_dir='/tmp/autoencoder')])



#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)

#encoder = load_model(ENCODER_FILE)
input_img = Input(shape=(784,))
encoded = autoencoder.layers[1](input_img)
encoded = autoencoder.layers[2](encoded)
encoded = autoencoder.layers[3](encoded)
encoder = Model(input_img, encoded)

encoded_imgs = encoder.predict(images)
#print(encoded_imgs.shape)

encoded_imgs = np.reshape(encoded_imgs, (encoded_imgs.shape[0], -1))
#kmeans = KMeans(n_clusters=2, random_state=0, verbose=1).fit(encoded_imgs)
#pickle.dump(kmeans, open(KMEANS_FILE, 'wb'))
kmeans = pickle.load(open(KMEANS_FILE, 'rb'))    
#print(kmeans.labels_)
#print(kmeans.cluster_centers_)
klabels = kmeans.labels_
#print("[1, 0] = [%d, %d]" % (np.sum(klabels), 140000-np.sum(klabels)))
#if abs(np.sum(klabels)-70000) < 200:
#    break

import pandas as pd
test = pd.read_csv(sys.argv[2], sep=',', encoding='latin-1', usecols=['ID','image1_index','image2_index'])

result_csv = open(sys.argv[3], 'w')
result_csv.write('ID,Ans\n')
for i in test['ID']:
    #klabels = kmeans.predict([encoded_imgs[test['image1_index'][i]], encoded_imgs[test['image2_index'][i]]])
    #if klabels[0] == klabels[1]:
    if klabels[test['image1_index'][i]] == klabels[test['image2_index'][i]]:
        result_csv.write(str(i)+',1\n')
    else:
        result_csv.write(str(i)+',0\n')
result_csv.close()
"""
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
"""
