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

def data_to_vector_MA1 (label_dir_celeba, number_labels, compressed_dataset):
    
    Y = np.array([pd.read_csv(label_dir_celeba, delimiter= '\t' )['gender'][:compressed_dataset.shape[0]]]).T
    Y = np.eye(number_labels)[Y.reshape(-1)]
    
    print('Shape of the gender_labels :', Y.shape)
    
    return Y

#=================================

def load_compressed_data():
    centroid_array = scipy.io.loadmat('dataset_array.mat')
    centroid_array = np.array(centroid_array['out'])
    
    idx_array = scipy.io.loadmat('idx_array.mat')
    idx_array = np.array(idx_array['out']) # expected shape (number of training example * number of features flatten)
    compressed_dataset = scipy.io.loadmat('reshapped_images.mat')
    compressed_dataset = np.array(compressed_dataset['out']) # expected shape (Number of images * compressed image set)
    
    return compressed_dataset, centroid_array, idx_array
    

#=================================

def get_data_MA1(number_labels):
    
    
    compressed_dataset, centroid_array, idx_array = load_compressed_data()
    X = compressed_dataset
    features = centroid_array
    Y = data_to_vector_MA1(label_dir_celeba, number_labels, compressed_dataset)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    print('Shape of features: ', centroid_array.shape)
    print('Shape of X_train:', X_train.shape)
    print('Shape of X_test:', X_test.shape)
    print('Shape of Y_train:', Y_train.shape)
    print('Shape of Y_test:', Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test, features


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

#=================================
# Random initiation of centroids

def kMeansInitCentroids(X, K):

    randidx = np.random.permutation(X.shape[0])
    initial_centroids = X[randidx[:K], :]
    
    print('Running ..')
    
    return initial_centroids

#=================================
def findClosestCentroids(X, initial_centroids):

    K = initial_centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(len(idx)):
        x = X[i]
        idx[i] = np.argmin([(x - v).dot(x - v) for v in initial_centroids])

    return idx

#==================================
def computeCentroids(X, idx, K):
    
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for i in range(K):
        centroids[i] = np.mean(X[idx == i], axis = 0)
     
    return centroids

#==================================
def runkMeans(X, centroids, findClosestCentroids, computeCentroids,
              max_iters, plot_progress=False):
    
    K = centroids.shape[0]

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)

    return centroids, idx

#===== Get Centroids  =================================================================


def get_centroids_celebaMA1(asset_dir_celeba, image_dir_celeba, samples,K,max_iters, target_size):
    
    centroid_array = []
    idx_array = []
    reshapped_images =[]
    

    for i in asset_dir_celeba[:int(samples)]:
        if i.endswith(".jpg"):
            
            img_sample = os.path.join(image_dir_celeba, i)
            img = image.img_to_array(image.load_img(img_sample, target_size = target_size, interpolation='bicubic'))
            img = img / 255
            img_array = img.reshape(-1, img.shape[2])
            
            # Initiate Centroids randomly
            initial_centroids = kMeansInitCentroids(img_array, K)
            
            # Iteratively calculate (i) Finding the closest centroids and (ii) Compute the new mean of the centroids as per the index data
            
            centroids, idx = runkMeans(img_array, initial_centroids, findClosestCentroids, computeCentroids, max_iters)
            if np.isnan(centroids).any() == True:
                replace_val = 0
                centroids[np.isnan(centroids)] = replace_val

            X_recovered = centroids[idx, :].reshape(img.shape)
            
            # Append data for future reference
            centroid_array.append(centroids)
            idx_array.append(idx)
            reshapped_images.append(X_recovered)
            
            # Saving the pre processing as a data to import in the future to avoid computational timing
            
    scipy.io.savemat('dataset_array.mat', mdict={'out' : centroid_array}, oned_as = 'column')
    scipy.io.savemat('idx_array.mat', mdict={'out': idx_array}, oned_as='column')
    scipy.io.savemat('reshapped_images.mat', mdict={'out': reshapped_images}, oned_as='column')

    #calculating mean of the centroids
    centroid_array = np.mean(centroid_array, axis=(0))

    return centroid_array, idx_array

#========MultiLayer Perceptron Model =================================================================

#=============== Initialize Parameters ========================

def initialize_parameters(X, Y, n_hidden_1, n_hidden_2):
    
    m = X.shape[0]
    n_h = X.shape[1] # compressed image height
    n_w = X.shape[2] # compressed image width
    n_c = X.shape[3] # compressed image channels
    n_y = Y.shape[1] # to investigate # number of classes
    
    
    X = v1.placeholder("float", [None, n_h, n_w, n_c])
    Y = v1.placeholder("float", [None, n_y])  # 2 output classes

    #images_flat = tf.reshape(features, [n_ce, n_ch])
    input_array = tf.reshape(X, [-1, X.shape[1] * X.shape[2] * X.shape[3]])
    #input_array = tf.cast(images_flat, tf.float32)

    stddev = 0.01
    
    weights = {
        'W1': tf .Variable(tf.random.normal([n_h * n_w * n_c , n_hidden_1], stddev=stddev)),
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



