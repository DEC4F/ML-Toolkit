#!usr/bin/python3
"""
This is an implementation of naive Bayes algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np

class LogisticRegression(object):
    """
    a naive bayes model
    """

    def __init__(self, n_bins, m_estimate,_lambda):
        self.n_bins = n_bins
        self.m_estimate = m_estimate
        self._lambda = _lambda
        self.weights = None
        self.lr = 0.001

    def fit(self, samples, labels):
        """
        build a naive bayes network with input samples and labels
        ----------
        samples : array-like
            the samples
        labels : array-like
            the labels
        """
        self.weights = np.zeros(samples.shape[1])
        params = np.dot(samples, self.weights)
        cond_likelihood = self.sigmoid(params)
        gradient = np.dot(samples.T, (cond_likelihood - labels )) / float(labels.size)
        self.weights += self.lr * gradient


    def predict(self, x):
        """
        predict the input instance's class label
        can only predict one sample at a time
        ----------
        X : array-like
            the sample data
        """
        pass

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
