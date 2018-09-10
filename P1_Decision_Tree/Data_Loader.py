#!/usr/bin/python3
"""
a generic data loader that loads the dataset provided by Dr. Soumya Ray
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import os
import numpy as np

class Loader(object):
    """
    load data
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.features, self.samples, self.labels = self.load()

    def load(self):
        """
        load and return attribute-value pairs
        ----------
        """
        data_name = self.file_path.split(os.sep)[-1]
        with open(os.path.join(self.file_path, data_name + '.names'), 'r') as _f:
            features = np.array([name.split(':')[0] for name in _f.readlines()])
            features = features[1:] # eliminate useless element

        with open(os.path.join(self.file_path, data_name + '.data'), 'r') as _f:
            samples = np.array([example.rstrip().split(',') for example in _f.readlines()])
            labels = np.array(list(map(int, samples[:, -1]))) # last column
            samples = samples[:, :-1] # first to second last column

        return features, samples, labels
