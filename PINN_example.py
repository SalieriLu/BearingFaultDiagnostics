#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri May  7 00:20:19 2021

@author: luhao
"""

import os
import scipy.io as scio
import numpy as np
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding,Add
from keras.layers import Conv1D, MaxPooling1D,Flatten,BatchNormalization,Reshape,Concatenate
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical,plot_model
# from keras.utils import to_categorical,plot_model
from keras import Input
import matplotlib.pyplot as plt
from keras.models import Model
from keras.callbacks import EarlyStopping

################################################
# Physical-based feature weighting (PFW) 

import keras 
import tensorflow as tf
class PWL(keras.layers.Layer):#Gaussian based weighting layer.
    def __init__(self, center_value=1,rate=1,name=None, **kwargs):
        super(PWL, self).__init__(name=name)
        self.center_value = center_value
        self.rate = rate
        super(PWL, self).__init__(**kwargs)
    def get_config(self):
        config = super(PWL,self).get_config()
      
        config["c_v"]=self.center_value
        config['rate']=self.rate
        config['k1'] = self.k1.numpy()
        config['sigma1']=self.sigma1.numpy()
        config['k2'] = self.k2.numpy()
        config['sigma2']=self.sigma2.numpy()
        config['k3'] = self.k3.numpy()
        config['sigma3']=self.sigma3.numpy()
        
        return config

    def build(self, input_shape):
        self.k1=self.add_weight( 
            shape=(1,),initializer='ones',trainable=True
            )
        self.k2=self.add_weight( 
            shape=(1,),initializer='ones',trainable=True
            )
        self.k3=self.add_weight( 
            shape=(1,),initializer='ones',trainable=True
            )
        self.sigma1 = self.add_weight(
            shape=(1,), initializer="ones", trainable=True
            )
        
        self.sigma2 = self.add_weight(
            shape=(1,), initializer="ones", trainable=True
            )
        self.sigma3 = self.add_weight(
            shape=(1,), initializer="ones", trainable=True
            )
        # super().build(input_shape)
        self.base_x = tf.reshape(tf.multiply(tf.range(int(input_shape[1]),dtype=tf.float32),0.01),[1,1600,1])
        self.base_x1 = tf.range(int(input_shape[1]),dtype=tf.float32)
        self.built = True 
    def call(self, inputs):
        # print(self.base_x1)
        # define the gaussian function

      
        location=tf.math.pow(self.k1,2)*tf.math.exp(-tf.math.pow((self.base_x-self.center_value),2)/2/tf.math.pow(tf.multiply(self.sigma1, 0.5),2))+\
        tf.math.pow(self.k2,2)*tf.math.exp(-tf.math.pow((self.base_x-self.center_value*2),2)/2/tf.math.pow(tf.multiply(self.sigma2, 0.5),2))+\
        tf.math.pow(self.k3,2)*tf.math.exp(-tf.math.pow((self.base_x-self.center_value*3),2)/2/tf.math.pow(tf.multiply(self.sigma3, 0.5),2))+1
        
        
        
        # print(location)
        return tf.multiply(inputs, location)




def shuflesignal(data):
    num_elements = data.shape[0] #
    new_index = np.arange(num_elements)  # generate index ranging from 0 to (num_elements-1)
    np.random.shuffle(new_index)  # shuffle the index
    data = data[new_index, :] 
    return data
new_sf=1000


file_path = r'C:\Users\luhao\Dropbox\Research_folder\Codes\CNN-CWL\processed_1_all'
file_path2 = r'C:\Users\luhao\Dropbox\Research_folder\Codes\CNN-CWL\processed_2_all'


# Construct training data for CNN model without PFW
os.chdir(file_path)
zz = scio.loadmat('result' + str(new_sf))
data0 = zz['outer_defect'] # Just a name but contain all five type faults
data1 = shuflesignal(data0)
x_train = data1[:, :1600]/100
y_train = data1[:, -1] # Labels 


# Construct test data
os.chdir(file_path2)
zz = scio.loadmat('result' + str(new_sf))
data0 = zz['outer_defect']
data1 = shuflesignal(data0)
x_test = data1[:,:1600]/100
y_test = data1[:,-1]

# Transfer the y label from int to categories.
y_train = to_categorical(y_train, num_classes = None)
y_test = to_categorical(y_test, num_classes = None)
classes = 5 # No fault(0); Ball fault(1); Inner race fault(2); Outer race fault(3); Combination fault(4)

# Construct the CNN model.
x0 = Input(batch_shape = (None, 1600, 1))
x01 = PWL(center_value=4.95)(x0)
x02 = PWL(center_value=3.048)(x0)
x03 = PWL(center_value=1.992)(x0)
x1 =  Concatenate(axis=1)([x0,x01,x02,x03])
# x1 = Add()([x0,x01,x02,x03])
x = Conv1D(8, 40, activation='relu', strides=5)(x1)
x = Conv1D(4, 20, padding='valid', activation='relu', strides=2)(x)
x = Conv1D(8, 10, padding='valid', activation='relu', strides=2)(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)
x = Dense(classes)(x)
x = Activation('softmax')(x)

# Train and validate CNN model.
model1 = Model(x0, x)

weight1=np.array([[1],[1],[1],[0.25],[0.25],[0.25]])
weight2=np.array([[1],[1],[1],[0.25],[0.25],[0.25]])
weight3=np.array([[1],[1],[1],[0.25],[0.25],[0.25]])

self_layer=model1.layers[1]
self_layer.set_weights(weight1)
self_layer=model1.layers[2]
self_layer.set_weights(weight2)
self_layer=model1.layers[3]
self_layer.set_weights(weight3)

model1.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


#Check model parameters
self_layer=model1.layers[1]
weights = self_layer.get_weights()
print(weights)


batch_size = 50
epochs = 150

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
history1 = model1.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,validation_split=0.1,verbose = 0, callbacks=[es])  # ,validation_data=[x_test,y_test])
# history1 = model1.fit(np.expand_dims(x_train, axis=2), y_train, batch_size = batch_size, epochs = epochs,verbose = 0)  # ,validation_data=[x_test,y_test])
loss,accu1 = model1.evaluate(np.expand_dims(x_test, axis=2), y_test)

print(accu1)

#Check model parameters
self_layer=model1.layers[1]
weights = self_layer.get_weights()
print(weights)

# #Another model

x0 = Input(batch_shape = (None, 1600, 1))
x = Conv1D(8, 40, activation='relu', strides=5)(x0)
x = Conv1D(4, 20, padding='valid', activation='relu', strides=2)(x)
x = Conv1D(8, 10, padding='valid', activation='relu', strides=2)(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)
x = Dense(classes)(x)
x = Activation('softmax')(x)

model2 = Model(x0, x)
model2.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# model.summary()
# plot_model(model,to_file='model.png')
# history=model.fit(np.expand_dims(x_train,axis=2), y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2)
batch_size = 50
epochs = 100


es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
history2 = model2.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,validation_split=0.1,verbose = 0, callbacks=[es])  # ,validation_data=[x_test,y_test])
# history = model2.fit(np.expand_dims(x_train, axis=2), y_train, batch_size = batch_size, epochs = epochs,verbose = 1)
loss,accu2 = model2.evaluate(np.expand_dims(x_test, axis=2), y_test) # ????? np.expand_dims

print(accu2)



x0 = Input(batch_shape = (None, 1600, 1))
x =  Concatenate(axis=1)([x0,x0,x0,x0])
# x = Add()([x0,x0,x0,x0])
x = Conv1D(8, 40, activation='relu', strides=5)(x)
x = Conv1D(4, 20, padding='valid', activation='relu', strides=2)(x)
x = Conv1D(8, 10, padding='valid', activation='relu', strides=2)(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)
x = Dense(classes)(x)
x = Activation('softmax')(x)

model3 = Model(x0, x)

model3.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# model.summary()
# plot_model(model,to_file='model.png')
# history=model.fit(np.expand_dims(x_train,axis=2), y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2)
batch_size = 50
# epochs = 100

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
history3 = model3.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,validation_split=0.1,verbose = 0, callbacks=[es])  # ,validation_data=[x_test,y_test])

loss,accu3 = model3.evaluate(np.expand_dims(x_test, axis=2), y_test) # ????? np.expand_dims

print(accu3)





for i in range(4):
    self_layer=model1.layers[i]
    weights = self_layer.get_weights()
    print(weights)



from sklearn.metrics import confusion_matrix
CM1=confusion_matrix(np.argmax(y_test,1), np.argmax(model1.predict(np.expand_dims(x_test, axis=2)),1))
CM2=confusion_matrix(np.argmax(y_test,1), np.argmax(model2.predict(np.expand_dims(x_test, axis=2)),1))
CM3=confusion_matrix(np.argmax(y_test,1), np.argmax(model3.predict(np.expand_dims(x_test, axis=2)),1))

print([accu1,accu2,accu3])
