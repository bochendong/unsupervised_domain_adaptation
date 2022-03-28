import numpy as np
import tensorflow as tf
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

import data_visualization
import mnist_m_gen


def to_gray_scale(x_train):
    new_xtrain = np.zeros([x_train.shape[0], 28, 28, 3], np.uint8)

    for i in range(x_train.shape[0]):
        for z in range(0, 3):
            for x in range(x_train.shape[1]):
                for y in range (x_train.shape[2]):
                    new_xtrain[i][y][x][z] = x_train[i][y][x]

    return new_xtrain

def get_numpy_data():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()

    minist_m_gen = mnist_m_gen.mnist_m_generator('./images/train/*.jpg', './images/test/*.jpg')
    minist_m_xtrain, minist_m_xtest = minist_m_gen.create_mnistm(x_train, x_test)

    new_x_train = to_gray_scale(x_train)
    new_x_test =  to_gray_scale(x_test)

    return new_x_train, new_x_test, minist_m_xtrain, minist_m_xtest, y_train, y_test
