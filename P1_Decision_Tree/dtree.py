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
    # parse args
    [use_full_sample, max_depth, use_gain_ratio] = [int(use_full_sample), int(max_depth), int(use_gain_ratio)]
    # parse dataset
    examples = np.array(mldata.parse_c45(file_path.split(os.sep)[-1], file_path), dtype=object)
    samples = examples[:, 1:-1]
    targets = examples[:, -1]
    # grow a huge tree (gurantees to cover a full tree) if input specifies 0 in max_depth
    if max_depth == 0:
        max_depth = int(1e9)
    # run on full sample
    if use_full_sample:
        dt = ID3(int(max_depth), use_gain_ratio)
        dt.fit(samples, targets)
    else:
        dt = ID3(max_depth, use_gain_ratio)
        # TODO: do CV on k-fold

if __name__ == '__main__':
    main()
