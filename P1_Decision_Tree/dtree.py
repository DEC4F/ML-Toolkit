#!usr/bin/python3
"""
This is an implementation of ID3 decision tree https://en.wikipedia.org/wiki/ID3_algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import sys, os
import mldata
from DT_Model import ID3

def main():
    """
    run the decision tree with param given by user
    """
    # file_path = '/home/decaf/Dropbox/EECS 440/dataset/voting'
    file_path, use_full_sample, max_depth, use_gain_ratio = sys.argv[1:]
    samples = mldata.parse_c45(file_path.split(os.sep)[-1], file_path)
    X = samples[:, 1:-1]
    y = samples[:, -1]
    print(y)
    #dt = ID3(max_depth, use_gain_ratio)
    #dt.fit(X, y)

if __name__ == '__main__':
    main()
