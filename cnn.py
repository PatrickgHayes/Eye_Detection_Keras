from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution2D, MaxPooling2D, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.layers import Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import cv2 as cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

#------------------------------------------------------------------------------
#                       Settings
#------------------------------------------------------------------------------
limit = 20
numEpochs = 5
num_classes = 6
sgd = SGD(lr=0.02, momentum=0.9, decay=0.001)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def loadImagesFromFolder(filename, encoding, perc_test):
    f_images = []
    f_labels = []
    for idx, img in enumerate(glob.glob(filename)):
        if idx > 20:
            break
        f_images.append(cv2.imread(img))
    numberOfImages = len(f_images)
    f_labels = np.array([encoding,]*numberOfImages)

    f_img_train = f_images[:len(f_images)/perc_test]
    f_img_test = f_images[len(f_images)/perc_test:]

    f_labels_train = f_labels[:f_labels.shape[0]/perc_test]
    f_labels_test = f_labels[f_labels.shape[0]/perc_test:]

    return f_img_train, f_img_test, f_labels_train, f_labels_test

def loadAndCat(img_train, img_test, labels_train, labels_test, filename, encoding, perc_test):
    f_img_train, f_img_test, f_labels_train, f_labels_test = loadImagesFromFolder(filename, encoding, perc_test) 
    img_train = np.vstack((img_train, f_img_train))
    img_test = np.vstack((img_test, f_img_test))
    labels_train = np.vstack((labels_train, f_labels_train))
    labels_test = np.vstack((labels_test, f_labels_test))

    return img_train, img_test, labels_train, labels_test

def loadAllImages(perc_test):
    img_train, img_test, labels_train, labels_test = loadImagesFromFolder(
                                            "/Users/patrickhayes/EyeDect/imgs/0_Eyes/* .jpeg",
                                            [0,0,0,0,0,1],perc_test) 

    img_train, img_test, labels_train, labels_test = loadAndCat(img_train, img_test, labels_train, labels_test, 
            "/Users/patrickhayes/EyeDect/imgs/2_Eyes/ *.jpeg", [0,0,0,1,0,0],perc_test) 
    img_train, img_test, labels_train, labels_test = loadAndCat(img_train, img_test, labels_train, labels_test,
            "/Users/patrickhayes/EyeDect/imgs/Bad_Angel/ *.jpeg", [0,1,0,0,0,0],perc_test) 
    img_train, img_test, labels_train, labels_test = loadAndCat(img_train, img_test, labels_train, labels_test, 
            "/Users/patrickhayes/EyeDect/imgs/Dead/ *.jpeg", [1,0,0,0,0,0],perc_test) 

    return img_train, img_test, labels_train, labels_test


#Loads the data into training set and test set
img_train, img_test, labels_train, labels_test = loadAllImages(2)

model = Sequential()

model.add(Convolution2D(5, 3, 3, border_mode='same', input_shape=(1024, 1280, 3), activation='relu'))
model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(img_train, labels_train, batch_size=10, nb_epoch=numEpochs,
                    verbose=1, callbacks=[], validation_data=(img_test, labels_test),
                    shuffle=True, class_weight='auto', sample_weight=None)
print history



