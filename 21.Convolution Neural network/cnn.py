#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:48:57 2017

@author: akhil
"""
from keras.models import Sequential
from keras.layers import Convolution2d
from keras.layers import MaxPooling2d
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()
classifier.add(Convolution2d(32,3,3,activation='relu',input_shape=(64,64,3)))
classifier.add(MaxPooling2d(poolsize=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_cross_entropy',metrics=['accuracy'])




