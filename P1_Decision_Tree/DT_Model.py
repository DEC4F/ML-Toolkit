#!usr/bin/python3
"""
This is an implementation of ID3 decision tree https://en.wikipedia.org/wiki/ID3_algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

class ID3(object):
    """
    a ID3 decision tree
    """

    def __init__(self, max_depth, criterion):
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X, y):
        pass
