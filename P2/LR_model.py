#!usr/bin/python3
"""
This is an implementation of naive Bayes algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np
import types

class LogisticRegression(object):
    """
    a naive bayes model
    """

    def __init__(self, learning_rate, num_iter, _lambda):
        self._lambda = _lambda
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.loss = None

    def fit(self, samples, labels):
        """
        build a Logistic Regression  with input samples and labels
        ----------
        samples : array-like
            the samples
        labels : array-like
            the labels
        """
        samples = samples.T
        for i, attrs in enumerate(samples):
            if not isinstance(attrs[0], (int, float)):
                samples[i] = self.encode(attrs)
        samples = samples.T

        self.weights = np.zeros(samples.shape[1])
        self.bias = 0
        for i in range(self.num_iter):
            gradient_weights, gradient_bias = self.gradient(samples, labels)
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, x):
        """
        predict the input instance's class label
        can only predict one sample at a time
        ----------
        X : array-like
            the sample data
        """
        pass

    def sigmoid(self, samples):
        '''
        logistic(sigmoid) function
        '''
        x = np.array(-(np.dot(samples, self.weights) + self.bias), dtype=np.float32)
        return 1.0 / (1 + np.exp(x))

    def encode(self, attrs):
        """
        encode nominal attributes to 1-k
        :param attrs: array-like
            attribute vector
        :return:
        """
        transdict = {}
        unique_vals = np.unique(attrs)
        for i, val in enumerate(unique_vals):
            transdict[val] = i
        return list(map(lambda x: transdict[x], attrs))

    def gradient(self, samples, labels):
        """
        :param samples: array-like
            Xs
        :param labels: array-like
            Ys
        :return: derivative_of_weights : array-like
            vector of weights' derivative
                derivative_of_bias : float
            derivative of bias
        """
        derivative_of_weights = np.zeros(samples.shape[1])
        derivative_of_bias = 0
        for i, X in enumerate(samples):
            cond_likelihood = self.sigmoid(X)
            if labels[i] == True:
                derivative_of_weights += (1 - cond_likelihood) * X
                derivative_of_bias += 1 - cond_likelihood
            else:
                derivative_of_weights += cond_likelihood * X
                derivative_of_bias += cond_likelihood

        derivative_of_weights += self._lambda * self.weights
        return derivative_of_weights, derivative_of_bias
