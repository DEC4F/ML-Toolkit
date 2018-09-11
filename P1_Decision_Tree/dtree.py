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
    samples = np.array(mldata.parse_c45(file_path.split(os.sep)[-1], file_path))
    X = samples[:, 1:-1]
    y = samples[:, -1]
    #dt = ID3(max_depth, use_gain_ratio)
    #dt.fit(X, y)

if __name__ == '__main__':
    main()
