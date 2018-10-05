#!usr/bin/python3
"""
This is an implementation of naive Bayes algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""
import sys
import os
import numpy as np
import mldata
from NB_Model import NaiveBayes

def main():
    """
    run the naive bayes network with given command line input
    ----------
    """
    file_path, use_full_sample, n_bins, m_estimate = sys.argv[1:5]
    # parse args
    [use_full_sample, n_bins, m_estimate] = [int(use_full_sample), int(n_bins), int(m_estimate)]
    examples = get_dataset(file_path)
    samples = examples[:, 1:-1]
    labels = examples[:, -1]
    n_b = NaiveBayes(n_bins, m_estimate)
    n_b.fit(samples, labels)

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

if __name__ == '__main__':
    main()
