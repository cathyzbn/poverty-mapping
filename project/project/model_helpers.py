import numpy as np
from scipy.special import expit as s_curve
from utils.logistic_regression_utils import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def predict_probability(data, weights):
    """
    Use the data and weights to calculate a probability for each data point.
    HINT: "dot products" can be done with np.dot(...)
    HINT: Use the sigmoid function which can be called with s_curve(...)
    HINT: data is of shape (dataset size, num features), and weights is 
    of shape (num features, 1)
    """
    pred = None
    ## YOUR CODE HERE
    val = np.dot(data, weights)
    pred = s_curve(val)
    ## END YOUR CODE
    return pred[...,None]

def sgd(data, labels, weights, learning_rate, regularization_rate):
    """
    Loop over all the data and labels, one at a time, and update the weights using the logistic
    regression learning rule.
    """
    for i in range(data.shape[0]):
        prob = predict_probability(data[i, :], weights)
        pi = predict_probability(data[i, :], weights)
        weights += learning_rate*data[i, :]*(labels[i] - pi)
        weights -= regularization_rate*weights
        
    return weights

def batch_sgd(data, labels, weights, learning_rate, regularization_rate, batch_size):
    """
    Loop over all the data and labels and update the weights using the logistic
    regression learning rule, averaged over multiple samples.
    HINT: This function will be very similar to "sgd", but you will need to use
    np.mean(...) to average up multiple gradients.
    """
    data_batch, labels_batch = create_batches(data, labels, batch_size)
    
    for ind, curr_batch in enumerate(data_batch):
        label_batch = labels_batch[ind]
        pi = predict_probability(curr_batch, weights)
        weights += np.mean(learning_rate*curr_batch*(label_batch-pi), axis=0)
        weights -= regularization_rate*weights
        ## END YOUR CODE

    return weights

def create_batches(data, labels, batch_size):
    data_batch = np.array_split(data, len(data)/batch_size)
    labels_batch = np.array_split(labels, len(labels)/batch_size)
    return data_batch, labels_batch

