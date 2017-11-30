
# coding: utf-8

from __future__ import print_function
import keras
from keras.datasets import cifar10, mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import RandomUniform
from keras.utils import plot_model
import os
import pickle
import numpy as np
import tensorflow as tf

from img_process import *
from lsuv_init import LSUVinit
import mnist_reader

flags = tf.app.flags
flags.DEFINE_integer("epochs", 25, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam[0.001] ")
flags.DEFINE_boolean("load_model", False, "loading model weights")
flags.DEFINE_boolean("multi_task", False, "using pretraning classifier weights")
flags.DEFINE_integer("datasets", 0, "0:using MNIST datasets,1:using fashionMNIST datasets,2:using notMNIST datasets")
FLAGS = flags.FLAGS
num_classes = 10
batch_size = 100

if(FLAGS.datasets == 0):
    data_class_name=['0','1','2','3','4','5','6','7','8','9']    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
if(FLAGS.datasets == 1):
    data_class_name=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
if(FLAGS.datasets == 2):
    data_class_name=['A','B','C','D','E','F','G','H','I','J']
    x_train, y_train = mnist_reader.load_mnist('data/notMNIST', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/notMNIST', kind='t10k') 

print('x_train shape:', x_train.shape)
x_train=re_scale(x_train)
x_test=re_scale(x_test)#重塑至[0,1]

x_train = np.reshape(x_train,[-1,28,28,1])
x_test = np.reshape(x_test,[-1,28,28,1])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train=i2b(x_train)
#x_test=i2b(x_test)#image to binary

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# The data, shuffled and split between train and test sets:
print('x_train shape:', x_train.shape)
print(y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(96, (3, 3), padding='same',input_shape=x_train.shape[1:]))
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

model.add(Conv2D(10, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.summary()
#plot_model(model, to_file='model.png')
if(FLAGS.multi_task==True):
    for i in range(27):
        w = np.load('./layer_weight/weights{}.00.npy'.format(i))
        if(i==26):
            w[0] = w[0][:,:,:,0:10]
            w[1] = w[1][0:10]
        model.layers[i].set_weights(w)


# train the model  
decay_rate = FLAGS.learning_rate / FLAGS.epochs
opt = keras.optimizers.Adam(lr=FLAGS.learning_rate,decay=decay_rate) 
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
if (FLAGS.load_model):
    model.load_weights('./checkpoint/learning_decay_weights.h5')
model.fit(x_train, y_train,batch_size,epochs=FLAGS.epochs,validation_data=(x_test, y_test),verbose=2)
if (FLAGS.load_model == False):
    model.save_weights('./checkpoint/learning_decay_weights.h5')
model.save_weights('./checkpoint/cnn_weights.h5')

result = model.predict(x_test)

matrix = np.zeros((10, 10))
right=0

for i, (Target, Label) in enumerate( zip(y_test, result) ) :  ### i-th label
    m = np.max(Label)
    for j, value in enumerate(Label) :  ### find max value and position
        if value == m :
            for k, num in enumerate(Target) :  ### find test calss
                if num == 1 :
                    matrix[k][j] += 1
                    break  # end of for k
            break  # end of for j

print('classification result')
for i in range(10):
    print('{}:{:.2f}% '.format(data_class_name[i],(matrix[i][i]/np.sum(matrix[i]))*100),end="")
    if i==4:print()
    right=right+matrix[i][i]

print("total:",len(result))
print("correct number：",right)
print("correct rate：{:.2f}%".format(right/len(result)*100))