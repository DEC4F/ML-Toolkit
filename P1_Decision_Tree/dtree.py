#!usr/bin/python3
"""
This is an implementation of ID3 decision tree https://en.wikipedia.org/wiki/ID3_algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import sys
from DT_Model import ID3
from Data_Loader import Loader

def main():
    """
    run the decision tree with param given by user
    """
    file_path, use_full_sample, max_depth, use_gain_ratio = sys.argv[1:]
    loader = Loader(file_path)
    X = loader.samples
    y = loader.labels
    print(y)    
    #dt = ID3(max_depth, use_gain_ratio)
    #dt.fit(X, y)

if __name__ == '__main__':
    main()
