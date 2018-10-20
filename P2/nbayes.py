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

K_FOLD = 5

def main():
    """
    run the naive bayes network with given command line input
    ----------
    """
    file_path, use_full_sample, n_bins, m_estimate = sys.argv[1:5]
    # parse args
    [use_full_sample, n_bins, m_estimate] = [int(use_full_sample), int(n_bins), int(m_estimate)]
    examples = get_dataset(file_path)
    np.random.shuffle(examples)
    n_b = NaiveBayes(n_bins, m_estimate)
    if use_full_sample:
        samples = examples[:, 1:-1]
        labels = examples[:, -1]
        n_b.fit(samples, labels)
    else:
        avg_vals, std = k_fold_cv(n_b, examples, K_FOLD)
        print ("Accuracy:  {:10f} {:10f}\nPrecision: {:10f} {:10f}\nRecall: {:13f} {:10f}".format(avg_vals[0], std[0],
                                                                                                  avg_vals[1], std[1],
                                                                                                  avg_vals[2], std[2]))


def get_dataset(file_path):
    """
    parse the dataset stored in the input file path
    ----------
    file_path : String
        the path to the dataset
    """
    raw_parsed = mldata.parse_c45(file_path.split(os.sep)[-1], file_path)
    return np.array(raw_parsed, dtype=object)


def precision(labels, predictions):
    """
    What fraction of the examples predicted positive are actually positive?
    """
    if sum(predictions) == 0:
        return 1.0
    #                  True              Positives / All positive predictions
    return sum(labels and predictions) / sum(predictions)


def recall(labels, predictions):
    """
    What fraction of the positive examples were predicted positive?
    """
    if sum(labels) == 0:
        return 1.0
    #                 True               Positives  / All positive labels
    return sum(labels and predictions) / sum(labels)


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
    prcisn = []
    rcll = []
    std =[]
    avg_vals = []
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


if __name__ == '__main__':
    main()
