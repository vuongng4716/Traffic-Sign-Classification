import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator

path = "C:\\Users\\ThinkPad\\Desktop\\myData"
labelFile = "C:\\Users\\ThinkPad\\Desktop\\labels.csv"

batch_size_val = 16
steps_per_epoch = 200
epochs = 10
imageDimensions = (32, 32, 3)
testRadio = 0.2
validationRatio = 0.2


count = 0
images = []
classNo = []
myList = []
myList = os.listdir(path)
noOfClasses = len(myList)

for x in range(0, len(myList)):
    myPicList = os.listdir(path+"\\"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"\\"+str(count)+"\\"+y)
        images.append(curImg)
        classNo.append(count)
    count += 1

images = np.array(images)
classNo = np.array(classNo)

#X_train, X_test, y_train, y_test = train_test_split(images, classNo,
                                                    #test_size=testRadio)
X_train, X_val, y_train, y_val = train_test_split(images, classNo,
                                                  test_size=testRadio)

data = pd.read_csv(labelFile)
num_of_samples = []
cols = 5
num_classes = noOfClasses
# fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
#X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# fig, axs = plt.subplots(1, 15, figsize=(20, 5))
# for i in range(15):
#     axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]))
#     axs[i].axis('off')
# plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)
#y_test = to_categorical(y_test, noOfClasses)

def MyModel():
    no_Of_filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_of_Nodes = 500

    model = Sequential()
    model.add((Conv2D(no_Of_filters, size_of_Filter, input_shape=(imageDimensions[0],
                                                                  imageDimensions[1], 1),
                      activation='relu')))
    model.add((Conv2D(no_Of_filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add((Conv2D(no_Of_filters//2, size_of_Filter2,
                      activation='relu')))
    model.add((Conv2D(no_Of_filters//2, size_of_Filter2,
                      activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(no_of_Nodes, activation='sigmoid'))
    model.add(Dense(120, activation='sigmoid'))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(RMSprop(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = MyModel()
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=32),
                              steps_per_epoch=len(X_train) // 32, epochs=10,
                              validation_data=(X_val, y_val), shuffle=1)



