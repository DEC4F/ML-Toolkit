#!usr/bin/python3
"""
This is an implementation of naive Bayes algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np
import math

class NaiveBayes(object):
    """
    a naive bayes model
    """

    def __init__(self, n_bins, m_estimate):
        self.n_bins = n_bins
        self.m_estimate = m_estimate

    def fit(self, samples, labels):
        """
        build a naive bayes network with input samples and labels
        ----------
        samples : array-like
            the samples
        labels : array-like
            the labels
        """
        pass

    def mean(self, numbers):
        """
        calculate the mean of the given attribute list
        ----------
        numbers : array-like
            the numbers

        """
        return sum(numbers) / float(len(numbers))

    def stdev(self, numbers):
        """
       calculate the standard deviation of the given attribute list
       ----------
       numbers : array-like
           the numbers

       """
        avg = self.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)

    def predict(self, x):
        """
        predict the input instance's class label
        can only predict one sample at a time
        ----------
        X : array-like
            the sample data
        """
        pass

    def discretize(self, cont_attr):
        """
        cont_attr : array-like
            continuous attribute (column of values) to be discretized
        n_bins : int
            partition the range of the feature into n bins
        """
        # generate evenly spaced list with n+1 values (n gaps/bins)
        bins = np.linspace(min(cont_attr), max(cont_attr), num=self.n_bins + 1)
        return np.digitize(cont_attr, bins)
