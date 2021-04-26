# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:44:30 2021

@author: HSingh
"""

#importing libraries

from keras.layers import Lambda, Dense, Input, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 

#resizing the image
img_size = [224,224]

trainpath = 'chest_xray/train'
testpath = 'chest_xray/test'

#preprocessing layer added in front of vgg16
#last layer will not be required and will be set to 2 categories
vgg = VGG16(input_shape = img_size + [3], weights = 'imagenet', include_top = False)

#weights of vgg16 are not trained
for layer in vgg.layers:
    layer.trainable = False

#getting the number of classes
folders = glob('chest_xray/train/*')

#adding layers
lyr = Flatten()(vgg.output)
pred = Dense(len(folders), activation='softmax')(lyr)

#creating model
model = Model(inputs = vgg.input, outputs = pred)

model.summary()

#assigning loss function and optimizer to the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#processing image
from keras.preprocessing.image import ImageDataGenerator

#setting image transformation for data augmentation
traindat_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
testdat_generator = ImageDataGenerator(rescale=1./255)

#training dataset and testing dataset
train_data = traindat_generator.flow_from_directory(trainpath, target_size = (224,224), 
                                                    batch_size = 32, class_mode = 'categorical')
test_data = testdat_generator.flow_from_directory(testpath, target_size = (224,224), 
                                                  batch_size = 32, class_mode = 'categorical')

#training the model
image_model = model.fit(train_data, validation_data=test_data, epochs = 3, 
                        steps_per_epoch=len(train_data), validation_steps=len(test_data))










    




