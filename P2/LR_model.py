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
        self.bias = None
        self.lr = 0.001

    def fit(self, samples, labels,num_iter = 10000):
        """
        build a Logistic Regression  with input samples and labels
        ----------
        samples : array-like
            the samples
        labels : array-like
            the labels
        """
        self.weights = np.zeros(samples.shape[1])
        self.weights, num_iter = grad_desc(samples, labels, weights)

        # params = np.dot(samples, self.weights)
        # cond_likelihood = self.sigmoid(params)
        # gradient = np.dot(samples.T, (cond_likelihood - labels )) / float(labels.size)
        # self.weights += self.lr * gradient
        # loss_log(cond_likelihood,labels)


    def predict(self, x):
        """
        predict the input instance's class label
        can only predict one sample at a time
        ----------
        X : array-like
            the sample data
        """
        pass
    def loss_log(self,samples,weights,labels):
        step1 = labels * np.log(self.sigmoid(samples,weights))
        step2 = (1 - labels) * np.log(1 - cond_likelihood)
        final = step1 + step2
        return np.mean(final)



    def log_gradient(self,weights, samples, labels):
        '''
        logistic gradient function
        '''
        first_calc = self.sigmoid(samples,weights) - labels.reshape(samples.shape[0], -1)
        final_calc = np.dot(first_calc.T, samples)
        return final_calc

    def sigmoid(self, samples,weights):
        '''
        logistic(sigmoid) function
        '''
        return 1.0 / (1 + np.exp(-np.dot(samples, weights.T)))

    def sigm(self, attrs):
        return 1.0 / (1 + np.exp(-np.dot(attrs, self.weights) + self.bias))

    def gradient(self, samples, labels):
        """
        :param samples:
        :param labels:
        :return:
        """
        dW = np.zeros(samples.shape[1])
        dB = 0
        for i, X in enumerate(samples):
            sigm = self.sigmoid(X)
            dW += -sigm * X
            dB += sigm
            if labels[i] == False:
                dX += X
                dB += 1
        dW += self._lambda * self.weights