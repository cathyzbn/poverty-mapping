import numpy as np
from project.models import *
from project.metrics import *
from utils.logistic_regression_utils import *
from itertools import *

def create_trucksplanes_dataset():

    """
    Creates a dataset from the trucks and planes dataset, using color
    histograms. You must slice the data and labels into a training and validation set.
    Try using roughly 10% of your training data for the validation!

    Try experimenting with different bins and values for use_hsv!
    """

    print("### Extracting Trucks and Planes Dataset ###")
    BINS = 30
    USE_HSV = False

    # features shape (dataset size, num features)
    features = extract_trucksplanes_histograms(BINS, USE_HSV)
    labels = load_trucksplanes_labels()
    features, labels = shuffle_data(features, labels)

    num_val = features.shape[0]//10
    num_train = features.shape[0] - num_val
    
    data_train = features[:num_train, :]
    data_validation = features[num_train:, :]
    labels_train = labels[:num_train]
    labels_validation = labels[num_train:]
    
    return data_train, labels_train, data_validation, labels_validation

def create_uganda_dataset():
    #10% dataset as validation
    features = extract_uganda_features()
    labels = load_satellite_labels()
    features, labels = shuffle_data(features, labels)

    num_val = features.shape[0]//10
    num_train = features.shape[0] - num_val

    data_train = features[:num_train, :]
    data_validation = features[num_train:, :]
    labels_train = labels[:num_train]
    labels_validation = labels[num_train:]
    
    return data_train, labels_train, data_validation, labels_validation

def cross_validation(use_satellite = False):
    best_hyperparameters = {
            "learning_rate": 0.0,
            "regularization_rate": 0.0,
            "batch_size": 1,
            "epochs":0
            }
    best_score_so_far = 0.0

    if use_satellite:
        data_train, labels_train, data_validation, labels_validation = create_uganda_dataset()
    else:
        data_train, labels_train, data_validation, labels_validation = create_trucksplanes_dataset()

    num_features = data_train.shape[1]

#     learning_rates = [0.0001]
#     regularizations = [0.125]
#     batch_sizes = [95]
#     epochs = [5]
# 77: lr:0.05 ; rr:1e-05 ; bs:60
# lr 0.1 : .71
    learning_rates = [0.01]
    regularizations = [0.00001]
    batch_sizes = [5]
    epochs = [500]

    print("### STARTING CROSS VALIDATION ###")
    for params in product(learning_rates, regularizations, batch_sizes, epochs):
        lr, reg, batch, epoch = params
        logreg_trainer = LogisticRegression(
            num_features, 
            lr, 
            reg,
            batch,
            epoch)

        logreg_trainer.train(data_train, labels_train)
        validationAccuracy = logreg_trainer.compute_accuracy(data_validation, labels_validation)
        if validationAccuracy > best_score_so_far:
            best_score_so_far = validationAccuracy
            best_hyperparameters = {
                "learning_rate": lr,
                "regularization_rate": reg,
                "batch_size": batch,
                "epochs": epoch
            }
        
    print("### FINISHED CROSS VALIDATION ###")
    print(best_hyperparameters)
    return best_hyperparameters

