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

QQ = 0


AUTOENCODER_FILE = 'hw6_model.h5'
KMEANS_FILE = 'kmeans.sav'
ENCODER_FILE = 'encoder.h5'
images = np.load(sys.argv[1])

# this is the size of our encoded representations
encoding_dim = 16  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# this is our input placeholder
input_img = Input(shape=(784,))
#input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
"""
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
"""

encoded = Dense(256, activation='sigmoid')(input_img)
encoded = Dense(64, activation='sigmoid')(encoded)
encoded = Dense(16, activation='sigmoid')(encoded)

decoded = Dense(64, activation='sigmoid')(encoded)
decoded = Dense(256, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

"""
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
encoded = Dense(128, activation='relu')(x)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
# 32 dimension (None, 32)


decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(3136, activation='relu')(decoded)
decoded = Reshape((14,14,16))(decoded)

#x = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (1, 1), activation='relu')(decoded)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
"""
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

autoencoder.summary()
# this model maps an input to its encoded representation
#encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
#encoded_input = Input(shape=(encoding_dim,))
encoded_input = Input(shape=(16,))
# retrieve the last layer of the autoencoder model

decoder_layer = autoencoder.layers[-3]
decoder_1 = decoder_layer(encoded_input)
decoder_layer = autoencoder.layers[-2]
decoder_2 = decoder_layer(decoder_1)
decoder_layer = autoencoder.layers[-1]
decoder_3 = decoder_layer(decoder_2)

"""
decoder_layer = autoencoder.layers[-7]
decoder_1 = decoder_layer(encoded_input)
decoder_layer = autoencoder.layers[-6]
decoder_2 = decoder_layer(decoder_1)
decoder_layer = autoencoder.layers[-5]
decoder_3 = decoder_layer(decoder_2)
decoder_layer = autoencoder.layers[-4]
decoder_4 = decoder_layer(decoder_3)
decoder_layer = autoencoder.layers[-3]
decoder_5 = decoder_layer(decoder_4)
decoder_layer = autoencoder.layers[-2]
decoder_6 = decoder_layer(decoder_5)
decoder_layer = autoencoder.layers[-1]
decoder_7 = decoder_layer(decoder_6)
"""
"""
decoder_layer = autoencoder.layers[-3]
decoder_8 = decoder_layer(decoder_7)
decoder_layer = autoencoder.layers[-2]
decoder_9 = decoder_layer(decoder_8)
decoder_layer = autoencoder.layers[-1]
decoder_10 = decoder_layer(decoder_9)
"""
# create the decoder model
decoder = Model(encoded_input, decoder_3)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
images = images.astype('float32') / 255.
images = np.reshape(images, (len(images), np.prod(images.shape[1:])))  # adapt this if using `channels_first` image data format

"""
images = images.reshape((len(images), np.prod(images.shape[1:])))
"""

x_train = images[:images.shape[0]*9//10]
x_test = images[images.shape[0]*9//10:]
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


#if QQ == 1:
#from keras.callbacks import TensorBoard
#es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
#cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', filepath=AUTOENCODER_FILE)

while(1):
    autoencoder.fit(images, images, epochs=5, batch_size=256, shuffle=True, verbose=1)
    autoencoder.save(AUTOENCODER_FILE)
    #encoder.save(ENCODER_FILE)
    del autoencoder
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
    print(encoded_imgs.shape)

    encoded_imgs = np.reshape(encoded_imgs, (encoded_imgs.shape[0], -1))
    kmeans = KMeans(n_clusters=2, random_state=0, verbose=1).fit(encoded_imgs)
    pickle.dump(kmeans, open(KMEANS_FILE, 'wb'))
    kmeans = pickle.load(open(KMEANS_FILE, 'rb'))    
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    klabels = kmeans.labels_
    print("[1, 0] = [%d, %d]" % (np.sum(klabels), 140000-np.sum(klabels)))
    if abs(np.sum(klabels)-70000) < 200:
        break

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
