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

def data_to_vector_MB1 (asset_dir_cartoonset, number_labels):
    
    if os.path.isdir(image_dir_cartoonset):
        X = []
        Y = []
    
        for i in asset_dir_cartoonset:
            if i.endswith(".png"):
                img_sample = os.path.join(image_dir_cartoonset, i)
                img_array = image.img_to_array(image.load_img(img_sample, target_size = (80,80), interpolation='bilinear'))
                img_array = img_array / 255
                X.append(img_array)
    
    X = np.array(X)
    Y = np.array([pd.read_csv(label_dir_cartoonset, delimiter= '\t' )['eye_color']]).T
    Y = np.eye(number_labels)[Y.reshape(-1)]
    
    X = X.reshape(X.shape[0],-1)
    Y = np.array(Y)
    
    print('Shape of the labels :', Y.shape)
    print('Shape of the dataset :', X.shape)
    
    return X, Y

#=====Data processing ========================================================

def split_data(X, Y):
    
    X_train = X[:6000] ; Y_train = Y[:6000]
    X_val = X[6000:8000] ; Y_val = Y[6000:8000]
    X_test = X[8000:] ; Y_test = Y[8000:]

    return X_train, X_test, Y_train, Y_test, X_val, Y_val

#=================================

def get_data_MB1(number_labels):
    
    X, Y = data_to_vector_MB1 (asset_dir_cartoonset, number_labels)
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
    s = 0.0001
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
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden fully connected layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['W2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer



#  Display data ==============================================

def displayData(X, figsize):

    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = 40 # or int(np.round(np.sqrt(n)))
    example_height= 40 # or int(np.round(n / example_width))

    # Compute number of items to display
    display_rows = 4
    display_cols = 4

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_height, example_width, 3, order='C'))
        ax.axis('off')