#!usr/bin/python3
"""
This is an implementation of ID3 decision tree https://en.wikipedia.org/wiki/ID3_algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import sys
import os
import numpy as np
import mldata
from DT_Model import ID3

def main():
    """
    run the decision tree with param given by user
    """
    file_path, use_full_sample, max_depth, use_gain_ratio = sys.argv[1:5]
    examples = np.array(mldata.parse_c45(file_path.split(os.sep)[-1], file_path), dtype=object)
    samples = examples[:, 1:-1]
    targets = examples[:, -1]
    dt = ID3(max_depth, use_gain_ratio)
    dt.fit(samples, targets)

def k_fold_cv(dt, data, k):
    data_split = np.array_split(data, k)
    acc = []
    for i in range(0, k):
        train_data = np.delete(data_split, (i), axis=0)
        train_data = np.concatenate(train_data)
        val_data = data_split[i]
        train_samples = train_data[:, 1:-1]
        train_targets = train_data[:, -1]
        val_samples = val_data[:, 1:-1]
        val_targets = [bool(x) for x in val_data[:, -1]]
        dt.fit(train_samples, train_targets)
        pred = [bool(dt.predict(val_samples[j, :])) for j in range(val_samples.shape[0])]
        acc.append(accuracy(val_targets, pred))
        print str(i + 1), " Fold Validated"
    return sum(acc) / float(k)

def accuracy(labels, predictions):
    assert len(labels) == len(predictions)
    count = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            count += 1
    return count / float(len(labels))

if __name__ == '__main__':
    main()
