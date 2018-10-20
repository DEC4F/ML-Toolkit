#!usr/bin/python3
"""
This is an implementation of several evaluation methods as an external library to be used to evaluate machine learning algorithms
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np

def precision(y_true, y_pred):
    """
    calculates the precision score of this prediction
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    if sum(y_pred) == 0:
        return 1.0
    # number of true positive / number of predicted positive
    return sum(y_true and y_pred) / sum(y_pred)

def recall(y_true, y_pred):
    """
    calculates the recall score of this prediction
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    if sum(y_true) == 0:
        return 1.0
    # number of true positive / number of factual true labels
    return sum(y_true and y_pred) / sum(y_true)

def accuracy(y_true, y_pred):
    """
    calculate the accuracy of prediction
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    assert len(y_true) == len(y_pred)
    count = 0
    for i, _ in enumerate(y_true):
        if y_true[i] == y_pred[i]:
            count += 1
    return count / float(len(y_true))

def k_fold_cv(model, data, k):
    """
    perform k fold cross validation on the model
    ----------
    model : the model produced by machine learning algorithm
          a model to be cross validated
    data : array-like
          the entire dataset
    k : int
          the parameter in cross validation determing how many fold we're doing
    """
    data_split = np.array_split(data, k)
    acc = []
    prcisn = []
    rcll = []
    std =[]
    avg_vals = []
    np.random.seed(12345)
    np.random.shuffle(data)

    for i in range(0, k):
        train_data = np.delete(data_split, (i), axis=0)
        train_data = np.concatenate(train_data)
        test_data = data_split[i]
        train_samples = train_data[:, 1:-1]
        train_targets = train_data[:, -1]
        test_samples = test_data[:, 1:-1]
        test_targets = [bool(x) for x in test_data[:, -1]]
        model.fit(train_samples, train_targets)
        pred = [bool(model.predict(test_samples[j, :])) for j in range(test_samples.shape[0])]
        acc.append(accuracy(test_targets, pred))
        prcisn.append(precision(test_targets, pred))
        rcll.append(recall(test_targets, pred))
    avg_vals = [sum(acc) / float(k) , sum(prcisn) / float(k) , sum(rcll) / float(k)]
    std = [np.std(acc) , np.std(prcisn) , np.std(rcll)]
    return avg_vals, std