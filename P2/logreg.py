#!usr/bin/python3
"""
This is an implementation of logistic regression algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""
import sys
import os
import numpy as np
import mldata
from LR_model import LogisticRegression


K_FOLD = 5

NUM_ITER = 10000
LEARNING_RATE = 0.01

def main():
    """
    run the logistic regression with given command line input
    ----------
    """
    file_path, use_full_sample, lbd = sys.argv[1:4]
    # parse args
    [use_full_sample, lbd] = [int(use_full_sample), int(lbd)]
    # parse dataset
    examples = get_dataset(file_path)
    log_reg = LogisticRegression(LEARNING_RATE, NUM_ITER, lbd)



    # run on full sample
    if use_full_sample:
        samples = examples[:, 1:-1]
        labels = examples[:, -1]
        log_reg.fit(samples, labels)
    else:
        print(k_fold_cv(log_reg, examples, K_FOLD))



def get_dataset(file_path):
    """
    parse the dataset stored in the input file path
    ----------
    file_path : String
        the path to the dataset
    """
    raw_parsed = mldata.parse_c45(file_path.split(os.sep)[-1], file_path)
    return np.array(raw_parsed, dtype=object)

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
    model : ID3
          an instance of ID3 to be cross validated
    data : array-like
          the entire dataset
    k : int
          the parameter in cross validation determing how many fold we're doing
    """
    data_split = np.array_split(data, k)
    acc = []
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
    return sum(acc) / float(k)

if __name__ == '__main__':
    main()
