#!usr/bin/python3
"""
This is an implementation of adaptive boosting algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np

class AdaBoosting(object):
    """
    a ada boosting classifier
    """

    def __init__(self, base_classifier, n_iter):
        self.n_iter = n_iter
        self.base_classifier = base_classifier
        self.classifiers = [base_classifier] * n_iter
        self.cls_weights = []

    def ensemble_fit(self, samples, labels, sample_weight=None):
        """
        Build a Bagging ensemble of classifier from the training
           set (X, y)
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted
        """
        # Initialize weights to 1/N
        N = labels.shape[0]
        weights = np.zeros(N)
        weights[:] = 1 / N
        for i in self.n_iter:
            # Train a classifier (after making it weights data compatible), and then classify the training data.
            #TODO: making learners weights data compatible
            self.classifiers[i].fit(samples, labels, weights = weights)
            predictions = self.classifiers[i].predict(samples)
            # Epsilon is the weighted training error.  Use it to calculate
            # alpha, the weight of this classifier in the vote.
            epsilon = np.sum(weights[predictions != labels])
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            if epsilon = 0 or epsilon ≥ 0.5:
                break
            # Store the classifiers weights.
            self.cls_weights.append(alpha)

            # Finally, update the weights of each example.
            new_weights = np.multiply(weights , np.exp(-(np.multiply(np.multiply(alpha * labels) * predictions))))
            weights = np.divide(new_weights , np.sum(new_weights))


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
