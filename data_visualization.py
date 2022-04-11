import matplotlib.pyplot as plt
import sklearn.manifold as manifold
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

import load_mnist_data
import numpy as np
'''
Usage:

python data_visualization.py --dataset mnist --dim 3
python data_visualization.py --dataset mnist --dim 2
python data_visualization.py --dataset mnist-m --dim 3
python data_visualization.py --dataset mnist-m --dim 2
'''

def tsne(x, dim, percentage, type):
     assert(percentage >0 and percentage < 1)
     assert(dim == 2 or dim == 3)
     
     tsne = manifold.TSNE(n_components=dim, random_state=42, learning_rate='auto', init='random')

     length = x.shape[0]
     sample_size = int(percentage * length)
     x = x[:sample_size]

     '''if (type == "mnist"):
          x.resize(x.shape[0], x.shape[1] * x.shape[1])
     else:'''

     x.resize(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
     transformed_data = tsne.fit_transform(x)
     return transformed_data

def plot_info(transformed_data, y, dim):
     xs, ys, zs = [], [], []
     color = []
     
     index = 0
     for index in range(transformed_data.shape[0]):
          xs.append(transformed_data[index][0])
          ys.append(transformed_data[index][1])
          if (dim == 3):
               zs.append(transformed_data[index][2])
          if (y[index] == 0): color.append('#1f77b4')
          elif (y[index] == 1): color.append('#ff7f0e')
          elif (y[index] == 2): color.append('#2ca02c')
          elif (y[index] == 3): color.append('#d62728')
          elif (y[index] == 4): color.append('#9467bd')
          elif (y[index] == 5): color.append('#8c564b')
          elif (y[index] == 6): color.append('#e377c2')
          elif (y[index] == 7): color.append('#7f7f7f')
          elif (y[index] == 8): color.append('#bcbd22')
          elif (y[index] == 9): color.append('#17becf')
          index += 1
    
     if (dim == 3):
          fig = plt.figure(figsize=(12,12))
          ax = fig.add_subplot(111, projection='3d')
          ax.scatter(xs,ys,zs, color = color)
     else:
          fig = plt.figure(figsize=(12,12))
          ax = fig.add_subplot(111)
          ax.scatter(xs,ys, color = color)

     plt.show()

def to_gray_scale(x_train):
    new_xtrain = np.zeros([x_train.shape[0], 28, 28, 3], np.uint8)

    for i in range(x_train.shape[0]):
        for z in range(0, 3):
            for x in range(x_train.shape[1]):
                for y in range (x_train.shape[2]):
                    new_xtrain[i][y][x][z] = x_train[i][y][x]

    return new_xtrain

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting')
    parser.add_argument('--dataset', default='minist', type=str, help='specify which dataset you want to check')
    parser.add_argument('--dim', default=3, type=int, help='project to which dimension')

    args = parser.parse_args()

    (x_train,y_train),(x_test,y_test) = mnist.load_data()
     
    X_train = to_gray_scale(x_train)

    if (args.dataset == 'mnist' and args.dim == 3):
        mnist_tranformed_data = tsne(X_train, 3, 0.1, type = "mnist")
        plot_info(mnist_tranformed_data, y_train, 3)
    elif (args.dataset == 'mnist' and args.dim == 2):
        mnist_tranformed_data = tsne(X_train, 2, 0.1, type = "mnist")
        plot_info(mnist_tranformed_data, y_train, 2)
    elif (args.dataset == 'mnist-m' and args.dim == 3):
        minist_m_gen = load_mnist_data.mnist_m_generator('./images/train/*.jpg', './images/test/*.jpg')
        minist_m_xtrain, minist_m_xtest = minist_m_gen.create_mnistm(x_train, x_test)

        mnist_m_tranformed_data = tsne(minist_m_xtrain, 3, 0.1,  type = "mnist-m")
        plot_info(mnist_m_tranformed_data, y_train, 3)
    elif(args.dataset == 'mnist-m' and args.dim == 2):
        minist_m_gen = load_mnist_data.mnist_m_generator('./images/train/*.jpg', './images/test/*.jpg')
        minist_m_xtrain, minist_m_xtest = minist_m_gen.create_mnistm(x_train, x_test)

        mnist_m_tranformed_data = tsne(minist_m_xtrain, 2, 0.1,  type = "mnist-m")
        plot_info(mnist_m_tranformed_data, y_train, 2)

