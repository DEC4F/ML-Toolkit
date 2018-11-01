#!usr/bin/python3
"""
This is an implementation of bagging algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np

class Bagging(object):
    """
    a bagging classifier
    """

    def __init__(self, base_classifier, n_iter):
        self.n_iter = n_iter
        self.base_classifier = base_classifier

    def ensemble_fit(self, samples, labels):
        """
        Build a Bagging ensemble of classifier from the training
           set (X, y) using boostrap
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values
        """
        pass

    def ensemble_predict(self, sample):
        """
        Predict class for X.
        The predicted class of an input sample is computed by majority vote
        ----------
        X : {array-like} of shape = [n_samples, n_features]
            The training input samples
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pass
