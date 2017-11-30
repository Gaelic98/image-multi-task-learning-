
# coding: utf-8

from __future__ import print_function
import keras
from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.initializers import RandomUniform
from keras.utils import plot_model
import os
import pickle
import numpy as np
import tensorflow as tf
import math

from img_process import *
from lsuv_init import LSUVinit
import mnist_reader
flags = tf.app.flags
flags.DEFINE_integer("epochs", 25, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam[0.001] ")
flags.DEFINE_boolean("load_model", False, "loading model weights")
FLAGS = flags.FLAGS

num_classes = 10
batch_size = 100
cy_train = np.zeros((60000,10))
cy_test = np.zeros((10000,10))

(mx_train, my_train), (mx_test, my_test) = mnist.load_data()
mx_train = np.asarray(mx_train)
mx_test = np.asarray(mx_test)
mx_train = np.reshape(mx_train,[-1,28,28,1])
mx_test = np.reshape(mx_test,[-1,28,28,1])
mx_train = mx_train.astype('float32')
mx_test = mx_test.astype('float32')
mx_train=re_scale(mx_train)
mx_test=re_scale(mx_test)

my_train = keras.utils.to_categorical(my_train, num_classes)
my_test = keras.utils.to_categorical(my_test, num_classes)
my_train = np.concatenate((my_train,cy_train),axis=1)
my_test = np.concatenate((my_test,cy_test),axis=1)

nx_train, ny_train = mnist_reader.load_mnist('data/notMNIST', kind='train')
nx_test, ny_test = mnist_reader.load_mnist('data/notMNIST', kind='t10k')
nx_train = np.asarray(nx_train)
nx_test = np.asarray(nx_test)
nx_train = np.reshape(nx_train,[-1,28,28,1])
nx_test = np.reshape(nx_test,[-1,28,28,1])
nx_train = nx_train.astype('float32')
nx_test = nx_test.astype('float32')
nx_train=re_scale(nx_train)
nx_test=re_scale(nx_test)

ny_train = keras.utils.to_categorical(ny_train, num_classes)
ny_test = keras.utils.to_categorical(ny_test, num_classes)
ny_train = np.concatenate((cy_train,ny_train),axis=1)
ny_test = np.concatenate((cy_test,ny_test),axis=1)
kx_train = np.concatenate((mx_train, nx_train),axis=0)
ky_train = np.concatenate((my_train, ny_train),axis=0)

print('x_train shape:', kx_train.shape)
print('y_train shape:', ky_train.shape)

model = Sequential()
model.add(Conv2D(96, (3, 3), padding='same',input_shape=kx_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3), padding='same',strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(192, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3), padding='same',strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(192, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(20, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

decay_rate = FLAGS.learning_rate / FLAGS.epochs
opt = keras.optimizers.Adam(lr=FLAGS.learning_rate,decay=decay_rate) 
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

if (FLAGS.load_model):
    model.load_weights('./checkpoint/learning_decay_weights.h5')
model.fit(kx_train,ky_train,batch_size=100,epochs=FLAGS.epochs,validation_data=(mx_test,my_test),verbose=2)
i = 0
for layer in model.layers:
    weights = layer.get_weights()
    np.save('./layer_weight/weights{:.2f}'.format(i),weights) 
    i +=1   
if (FLAGS.load_model == False):
    model.save_weights('./checkpoint/learning_decay_weights.h5')
model.save_weights('./checkpoint/learning_decay_weights.h5')