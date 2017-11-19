import sys
import numpy as np
import keras
import os
from keras.models import Model, load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adadelta
from keras.utils import np_utils
import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))

train_csv = open(sys.argv[1], "r")
idx = 0
train_feature = []
train_label = []
train_label_count = [0]*7
for r in train_csv:
    if idx != 0:
        t = r.split(",")
        t[0] = int(t[0])
        tmp = [0]*7
        tmp[t[0]] = 1
        train_label.append(t[0])
        train_label_count[t[0]] += 1
        t[1] = t[1].split()
        t[1] = list(map(lambda x:int(x), t[1]))
        train_feature.append(t[1])
    idx += 1

train_csv.close()

batch_size = 128
nb_classes = 7
nb_epoch = 100

# input image dimensions
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

train_feature = np.array(train_feature)
train_label = np.array(train_label)

#normalize
train_feature_mean = np.mean(train_feature, 0)
train_feature_std = np.std(train_feature, 0)
train_feature = (train_feature-train_feature_mean)/train_feature_std

train_feature = train_feature.reshape(train_feature.shape[0], img_rows, img_cols, 1)
#train_feature = train_feature.reshape(train_feature.shape[0], 2304, 1)
train_feature = train_feature.astype("float32")

training_feature_set = train_feature[:(len(train_feature)*4)//5]
validation_feature_set = train_feature[(len(train_feature)*4)//5:]

training_feature_set = training_feature_set.reshape(training_feature_set.shape[0], img_rows, img_cols, 1)
validation_feature_set = validation_feature_set.reshape(validation_feature_set.shape[0], img_rows, img_cols, 1)
training_feature_set = training_feature_set.astype("float32")
validation_feature_set = validation_feature_set.astype("float32")
#training_feature_set /= 255
#validation_feature_set /= 255

train_label_tmp = train_label

train_label = np_utils.to_categorical(train_label, nb_classes)


training_label_set = train_label[:(len(train_label)*4)//5]
validation_label_set = train_label[(len(train_label)*4)//5:]
validation_label_set_tmp = train_label_tmp[(len(train_label_tmp)*4)//5:]

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(48, 48, 1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (5, 5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))

model.add(Conv2D(64, (5, 5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))


model.add(Conv2D(128, (5, 5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(128, (5, 5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))

model.add(Conv2D(256, (5, 5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(256, (5, 5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))


model.add(Flatten())
model.add(Dense(units=1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(units=1024))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Dense(7))
model.add(BatchNormalization())
model.add(Activation('softmax'))

#ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=False)

datagen.fit(training_feature_set)





for e in range(nb_epoch):
    print(e)
    batches = 0
    for X_batch, Y_batch in datagen.flow(train_feature, train_label, batch_size=batch_size):
        model.train_on_batch(X_batch, Y_batch)
        batches += 1
        print(batches)
        if batches >= len(train_feature) / batch_size:
            print("batches break")
            break
    model.save('model-%d.h5' %(e+1))

