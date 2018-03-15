#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:09:08 2018

@author: rakib
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

batch_size = 5000
num_classes = 10
epochs = 1
epochs_part=1
repeat=2
# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_temp=[]
x_temp=[]
class_len=[]
print(y_train.shape)
print(x_train.shape)
for i in range (0,num_classes):
	k=0
	for j in range (0,len(y_train)):
		if y_train[j]==i:
			y_temp.append(y_train[j])
			x_temp.append(x_train[j,:,:])	
			k=k+1
	class_len.append(k)
	#print(k)



x_train_old=x_train
y_train_old=y_train
y_train=np.array(y_temp)
x_train=np.array(x_temp)
print(x_train.shape)
print(y_train.shape)
class_len=np.array(class_len)
print(class_len)

y_temp=[]
x_temp=[]
test_class_len=[]
print(y_test.shape)
print(x_test.shape)
for i in range (0,num_classes):
	k=0
	for j in range (0,len(y_test)):
		if y_test[j]==i:
			y_temp.append(y_test[j])
			x_temp.append(x_test[j,:,:])	
			k=k+1
	test_class_len.append(k)
	#print(k)

x_test_old=x_test
y_test_old=y_test
x_test=np.array(x_temp)
y_test=np.array(y_temp)
print(x_test.shape)
test_class_len=np.array(test_class_len)
print(test_class_len)
#np.savez(mnist,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)

		

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_train_old = x_train_old.reshape(x_train_old.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train_old = x_train_old.reshape(x_train_old.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train_old = x_train_old.astype('float32')

x_train /= 255
x_train_old /=255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
ytrain_old=y_train_old
y_train_old = keras.utils.to_categorical(y_train_old, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
train_len=len(y_train_old)
train_acc_full=[]
train_loss_full=[]
val_acc_full=[]
val_loss_full=[]
l=0
test_acc_full=np.zeros((num_classes,epochs*train_len/batch_size))
for epoch in range(0,epochs):
        k=0
        for i in range (0,train_len/batch_size):
                hist=model.fit(x_train_old[k:k+batch_size], y_train_old[k:k+batch_size],batch_size=batch_size,epochs=1,verbose=1,validation_data=(x_test, y_test), shuffle=True)
                train_acc_full.append(hist.history['acc'])
                train_loss_full.append(hist.history['loss'])
                val_acc_full.append(hist.history['val_acc'])
                val_loss_full.append(hist.history['val_loss'])
                # Classwise
                m=0
                for j in range (0,num_classes):
                        score = model.evaluate(x_test[m:m+test_class_len[j]],y_test[m:m+test_class_len[j]], verbose=0)
                        #print(score[1])
                        test_acc_full[j,l]=score[1]
                        m=m+test_class_len[j]
                l=l+1	
                k=k+batch_size

train_acc_full=np.array(train_acc_full)
train_loss_full=np.array(train_loss_full)
val_acc_full=np.array(val_acc_full)
val_loss_full=np.array(val_loss_full)

# Now Removing Begins

train_acc_part=[]
train_loss_part=[]
val_acc_part=[]
val_loss_part=[]
l=0
test_acc_part=np.zeros((num_classes,epochs_part*num_classes*repeat*np.int(np.floor((60000-6000)/batch_size)+1)))

for epoch in range(0,epochs_part):
        for num in range (0,num_classes):
                y_temp=[]
                x_temp=[]
                for j in range (0,len(ytrain_old)):
                        if ytrain_old[j]!=num:
                                y_temp.append(ytrain_old[j])
                                x_temp.append(x_train_old[j,:,:])	
                x_temp=np.array(x_temp)
                y_temp=np.array(y_temp)
                y_temp = keras.utils.to_categorical(y_temp, num_classes)
                # Randomize
                temp_len=len(y_temp)
                print(x_temp.shape)
                print(temp_len)
                temp=np.arange(temp_len)
                np.random.shuffle(temp)
                for rep in range (0,repeat):
                        print(np.int(np.floor(temp_len/batch_size)+1))
                        k=0
                        for i in range (0,np.int(np.floor(temp_len/batch_size)+1)):
                                
                                hist=model.fit(x_train[k:np.minimum(k+batch_size,temp_len)], y_train[k:np.minimum(k+batch_size,temp_len)],batch_size=np.minimum(batch_size,temp_len-k+1),epochs=1,verbose=1,validation_data=(x_test, y_test), shuffle=True)
                                train_acc_part.append(hist.history['acc'])
                                train_loss_part.append(hist.history['loss'])
                                val_acc_part.append(hist.history['val_acc'])
                                val_loss_part.append(hist.history['val_loss'])
                                # Classwise
                                m=0
                                for j in range (0,num_classes):
                                        score = model.evaluate(x_test[m:m+test_class_len[j]],y_test[m:m+test_class_len[j]], verbose=0)
                                        #print(score[1])
                                        test_acc_part[j,l]=score[1]
                                        m=m+test_class_len[j]
                                l=l+1	
                                k=k+batch_size
                                
                
train_acc_part=np.array(train_acc_part)
train_loss_part=np.array(train_loss_part)
val_acc_part=np.array(val_acc_part)
val_loss_part=np.array(val_loss_part)                        

#print(test_acc)
np.savez('mnist_result_gradual'+str(epochs),test_acc_full=test_acc_full,train_acc_full=train_acc_full,train_loss_full=train_loss_full,val_acc_full=val_acc_full,val_loss_full=val_loss_full,test_acc_part=test_acc_part,train_acc_part=train_acc_part,train_loss_part=train_loss_part,val_acc_part=val_acc_part,val_loss_part=val_loss_part)
plt.plot(val_acc_full)
plt.ylabel('val_acc')
plt.show()
plt.plot(val_loss_full)
plt.ylabel('val_loss')
plt.show()
plt.plot(train_acc_full)
plt.ylabel('train_acc')
plt.show()
plt.plot(train_loss_full)
plt.ylabel('train_loss')
plt.show()
plt.plot(val_acc_part)
plt.ylabel('val_acc')
plt.show()
plt.plot(val_loss_part)
plt.ylabel('val_loss')
plt.show()
plt.plot(train_acc_part)
plt.ylabel('train_acc')
plt.show()
plt.plot(train_loss_part)
plt.ylabel('train_loss')
plt.show()
'''
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
