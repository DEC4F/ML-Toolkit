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

    def __init__(self,lr,num_iter,_lambda):
        self.bias = None
        self._lambda = _lambda
        self.weights = None
        self.bias = None
        self.lr = lr
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
        self.bias = np.zeros(samples.shape[1])
        self.weights = np.zeros(samples.shape[1])
        for i in range(samples.shape[1]):
            samples[:, i] = self.encode(samples[:, i])
        labels = self.encode(labels)
        cost = self.LR_likelihood(samples,labels)
        while (self.num_iter):
            old_cost = cost
            self.weights = self.weights - (self.lr * self.log_gradient( samples, labels))
            self.bias = self.bias - (self.lr * self.log_gradient(samples, labels))
            cost = self.LR_likelihood(samples,labels)
            self.loss = old_cost - cost
            self.num_iter -= 1


    def predict(self, x):
        """
        predict the input instance's class label
        can only predict one sample at a time
        ----------
        X : array-like
            the sample data
        """
        pass



    def log_gradient(self, X, y):
        '''
        logistic gradient function
        '''
        log_grad = np.dot(self.sigmoid(X) - y.reshape(X.shape[0], -1).T, X)
        return log_grad

    def LR_likelihood(self, samples, labels, _lambda=0.1):
        m = len(labels)
        h = self.sigmoid(samples)
        reg = (_lambda / (2 * m)) * np.sum(self.weights ** 2)

        return (1 / m) * (-labels.T.dot(np.log(h)) - (1 - labels).T.dot(np.log(1 - h))) + reg

    def sigmoid(self,samples):
        '''
        logistic(sigmoid) function
        '''
        x = np.array(-(np.dot(samples, self.weights.T + self.bias)), dtype=np.float32)
        return 1.0 / (1 + np.exp(x))


    def encode(self, samples):
        if not isinstance(samples[0], (int, float)) or type(samples[0]) == types.BooleanType:
            transdict = {}
            for x in range(0, np.unique(samples).shape[0]):
                transdict[np.unique(samples)[x]] = x
            idx = np.nonzero(transdict.keys() == samples[:, None])[1]
            samples = np.asarray(transdict.values())[idx]
        return samples

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
