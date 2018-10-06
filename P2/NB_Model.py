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

    def mean(self, numbers):
        """
        calculate the mean of the given attribute list
        ----------
        numbers : array-like
            the numbers

        """
        return sum(numbers) / float(len(numbers))

        pass

    def stdev(self, numbers):
        """
       calculate the standard deviation of the given attribute list
       ----------
       numbers : array-like
           the numbers

       """
        avg = mean(numbers)
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

    def discretize(self, numbers, k):
        """

        :param numbers:
            array-like, continuous values to be discretized
        :param k: partition the range of the feature into k bins
        :return:
        """

        min = np.min(numbers)
        bin_size = (np.max(numbers) - min) / k

        for i in range(len(numbers)):
            bin_number = 0
            upper_bound = min + bin_size

            # search the bin to which numbers[i] belongs
            while numbers[i] > upper_bound:
                bin_number += 1
                upper_bound += bin_size

            # replace the feature with the bin number
            numbers[i] = bin_number
