import numpy as np
import tensorflow as tf
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from torch.utils.data import TensorDataset, DataLoader

import torch

import numpy as np
import tensorflow as tf
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

'''
This class will take the path of BSDS data set as input and 
use them as background to generte MNIST-m data set

Usage:
(x_train,y_train),(x_test,y_test) = mnist.load_data()
minist_m_gen = mnist_m_gen.mnist_m_generator('./images/train/*.jpg', './images/test/*.jpg')
minist_m_xtrain, minist_m_xtest = minist_m_gen.create_mnistm(x_train, x_test)
'''

class mnist_m_generator(object):
    def __init__(self, train_dir, test_dir):
        self.train_filelist = glob.glob(train_dir)
        self.test_filelist = glob.glob(test_dir)
        self.DS500_train_dataset= []
        self.DS500_test_dataset = []

        for fname in self.train_filelist:
            im = Image.open(fname)
            im = np.array(im.resize((321, 481)))

            self.DS500_train_dataset.append(im)

        for fname in self.test_filelist:
            im = Image.open(fname)
            im = np.array(im.resize((321, 481)))

            self.DS500_test_dataset.append(im)

        self.DS500_train_dataset = np.array(self.DS500_train_dataset)
        self.DS500_test_dataset = np.array(self.DS500_test_dataset)

    def create_mnistm(self, X_train, X_test):
        def compose_image(mnist_data, background_data):
            w, h, _ = background_data.shape
            dw, dh, _ = mnist_data.shape
            x = np.random.randint(0, w - dw)
            y = np.random.randint(0, h - dh)
            bg = background_data[x:x + dw, y:y + dh]
            return np.abs(bg - mnist_data).astype(np.uint8)

        def mnist_to_img(x):
            x = (x > 0).astype(np.float32)
            d = x.reshape([28, 28, 1]) * 255
            return np.concatenate([d, d, d], 2)

        minist_m_xtrain = np.zeros([X_train.shape[0], 28, 28, 3], np.uint8)
        for i in range(X_train.shape[0]):
            index = np.random.choice(self.DS500_train_dataset.shape[0], 1, replace=False)
            bg_img = self.DS500_train_dataset[index][0]
            
            mnist_image = mnist_to_img(X_train[i])

            mnist_image = compose_image(mnist_image, bg_img)
            minist_m_xtrain[i] = mnist_image

        minist_m_xtest = np.zeros([X_test.shape[0], 28, 28, 3], np.uint8)
        for i in range(X_test.shape[0]):
            index = np.random.choice(self.DS500_test_dataset.shape[0], 1, replace=False)

            bg_img = self.DS500_test_dataset[index][0]
            mnist_image = mnist_to_img(X_test[i])

            mnist_image = compose_image(mnist_image, bg_img)
            minist_m_xtest[i] = mnist_image

        return minist_m_xtrain, minist_m_xtest


def to_gray_scale(x_train):
    new_xtrain = np.zeros([x_train.shape[0], 28, 28, 3], np.uint8)

    for i in range(x_train.shape[0]):
        for z in range(0, 3):
            for x in range(x_train.shape[1]):
                for y in range (x_train.shape[2]):
                    new_xtrain[i][y][x][z] = x_train[i][y][x]

    return new_xtrain

def get_numpy_data(percentage):
    (x_train,y_train),(x_test,y_test) = mnist.load_data()

    total = int(x_train.shape[0] * percentage)

    x_train = x_train[:total]
    y_train = y_train[:total]


    minist_m_gen = mnist_m_generator('./images/train/*.jpg', './images/test/*.jpg')
    minist_m_xtrain, minist_m_xtest = minist_m_gen.create_mnistm(x_train, x_test)

    new_x_train = to_gray_scale(x_train)
    new_x_test =  to_gray_scale(x_test)

    new_x_train = (new_x_train/127.5) - 1
    new_x_test = (new_x_test/127.5) - 1
    minist_m_xtrain = (minist_m_xtrain/127.5) - 1
    minist_m_xtest = (minist_m_xtest/127.5) - 1

    return new_x_train, new_x_test, minist_m_xtrain, minist_m_xtest, y_train, y_test

def get_data_loader(percentage):

    x_train, x_test, mnist_m_x_train, mnist_m_x_test, y_train, y_test = get_numpy_data(percentage)


    x_train = x_train.reshape((x_train.shape[0], 3, 28, 28))
    x_test = x_test.reshape((x_test.shape[0], 3, 28, 28))
    mnist_m_x_train = mnist_m_x_train.reshape((mnist_m_x_train.shape[0], 3, 28, 28))
    mnist_m_x_test = mnist_m_x_test.reshape((mnist_m_x_test.shape[0], 3, 28, 28))
    
    mnist_X_train = torch.Tensor(x_train)
    mnist_X_test =  torch.Tensor(x_test)
    mnist_m_X_train = torch.Tensor(mnist_m_x_train)
    mnist_m_X_test = torch.Tensor(mnist_m_x_test)
    train_label = torch.Tensor(y_train)
    test_label = torch.Tensor(y_test)

    kwargs = {}

    mnist_dataset = TensorDataset(mnist_X_train, train_label)
    dataloader_source = DataLoader(mnist_dataset, shuffle = True, batch_size=128, **kwargs)

    mnist_m_dataset = TensorDataset(mnist_m_X_train,train_label)
    dataloader_target = DataLoader(mnist_m_dataset, shuffle = True, batch_size=128, **kwargs)

    return dataloader_source, dataloader_target
