import tensorflow as tf
import tensorflow.compat.v1 as v1
import numpy as np
import os
from keras.preprocessing import image
import scipy.io
import pandas as pd
from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
from sklearn.model_selection import train_test_split


data_dir = ('Datasets')
image_dir_celeba = os.path.join(data_dir, 'celeba', 'img')
image_dir_cartoonset = os.path.join(data_dir,'cartoon_set', 'img')
asset_dir_celeba = os.listdir(os.path.join(os.getcwd(), 'Datasets/celeba/img'))
asset_dir_cartoonset = os.listdir(os.path.join(os.getcwd(), 'Datasets/cartoon_set/img'))
label_dir_celeba = os.path.join(data_dir,'celeba', 'labels.csv')
label_dir_cartoonset = os.path.join(data_dir, 'cartoon_set', 'labels.csv')

#=================================

def data_to_vector_MA2 (asset_dir_celeba, number_labels):
    
    if os.path.isdir(image_dir_celeba):
        X = []
        Y = []
    
        for i in asset_dir_celeba:
            if i.endswith(".jpg"):
                img_sample = os.path.join(image_dir_celeba, i)
                img_array = image.img_to_array(image.load_img(img_sample, target_size = (79, 59), interpolation='bicubic'))
                img_array = img_array / 255
                X.append(img_array)
    
    X = np.array(X)
    Y = np.array([pd.read_csv(label_dir_celeba, delimiter= '\t' )['smiling']]).T
    Y = np.eye(number_labels)[Y.reshape(-1)]
    
    X = X.reshape(X.shape[0],-1)
    Y = np.array(Y)
    
    print('Shape of the gender_labels :', Y.shape)
    print('Shape of the image dataset :', X.shape)
    
    return X, Y

#=====Data processing ========================================================

def split_data(X, Y):
    
    X_train = X[:3000] ; Y_train = Y[:3000]
    X_val = X[3000:4000] ; Y_val = Y[3000:4000]
    X_test = X[4000:] ; Y_test = Y[4000:]

    return X_train, X_test, Y_train, Y_test, X_val, Y_val

#=================================

def get_data_MB1(number_labels):
    
    X, Y = data_to_vector_MB1 (asset_dir_celeba, number_labels)
    X_train, X_test, Y_train, Y_test, X_val, Y_val = split_data(X, Y)
    
    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)
    print('Shape of X_train:', X_train.shape)
    print('Shape of X_test:', X_test.shape)
    print('Shape of Y_train:', Y_train.shape)
    print('Shape of Y_test:', Y_test.shape)
    print('Shape of X_val:', X_val.shape)
    print('Shape of Y_val:', Y_val.shape)
    
    return X_train, X_test, Y_train, Y_test, X_val, Y_val

#=====PCA Compression ========================================================

# Get the eigen value and eigen vector

def PCA_compression(X, K): # mention of K depending on the image resolution and criteria

    X_norm, mu, sigma = featureNormalize(X)
    U, S = pca(X_norm)
    Z = projectData(X_norm, U, K)
    
    print('The projected data Z has a shape of: ', Z.shape)
    return Z, U, S, X_norm

def pca(X):

    m, n = X.shape

    # value initialization
    U = np.zeros(n)
    S = np.zeros(n)
    
    Sigma = (np.dot(X.T, X)) / X.shape[0]
    U, S, V = np.linalg.svd(Sigma)

    return U, S

def featureNormalize(X):

    mu = np.mean(X, axis=0)
    
    # Correction of Zeros - Sigma
    X_norm = X - mu
    s = np.min(X_norm[X_norm != 0])
    X_norm[X_norm == 0] = s 
    
    # Correction of Zeros - Sigma
    sigma = np.std(X_norm, axis=0)
    s = 1
    sigma[sigma == 0] = s
    
    
    X_norm = np.divide(X_norm, sigma)
    
    return X_norm, mu, sigma

def projectData(X, U, K):
    
   # takes in U (array-like dim: n x n) - The computed eigenvectors using PCA. Each column in matrix is a single eigenvector
   # Returns Z (array-like dim: m x k): projection of top eigenvectors 

    Ureduce = U[:, :K]
    Z = np.dot(X, Ureduce)

    return Z

#========MultiLayer Perceptron Model =================================================================

#=============== Initialize Parameters ========================

def initialize_parameters(X, Y, n_hidden_1, n_hidden_2):
    
    m = X.shape[0]
    n_f = X.shape[1]
    n_y = Y.shape[1] # to investigate # number of classes
    
    
    X = v1.placeholder("float", [None, n_f])
    Y = v1.placeholder("float", [None, n_y])  # 2 output classes

    #images_flat = tf.reshape(features, [n_ce, n_ch])
    input_array = tf.reshape(X, [-1, X.shape[1]])
    #input_array = tf.cast(images_flat, tf.float32)

    stddev = 0.01
    
    weights = {
        'W1': tf .Variable(tf.random.normal([n_f , n_hidden_1], stddev=stddev)),
        'W2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random.normal([n_hidden_2, n_y], stddev=stddev))
    }

    biases = {
        'b1': tf.Variable(tf.random.normal([n_hidden_1], stddev=stddev)),
        'b2': tf.Variable(tf.random.normal([n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random.normal([n_y], stddev=stddev))
    }

    return weights, X, Y, biases, input_array

#===============Forward Propagation ========================



def multilayer_perceptron(weights, biases, input_array):

    # Hidden fully connected layer 1
    layer_1 = tf.add(tf.matmul(input_array, weights['W1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)

    # Hidden fully connected layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['W2']), biases['b2'])
    layer_2 = tf.sigmoid(layer_2)
    
    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer



#  Display data ==============================================

