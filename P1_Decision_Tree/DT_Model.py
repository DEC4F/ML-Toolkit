#!usr/bin/python3
"""
This is an implementation of ID3 decision tree https://en.wikipedia.org/wiki/ID3_algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np

class ID3(object):
    """
    a ID3 decision tree
    """

    def __init__(self, max_depth, use_gain_ratio):
        self.max_depth = max_depth
        self.use_gain_ratio = use_gain_ratio

    def fit(self, samples, labels):
        """
        build a id3 decision tree with input samples and labels
        ---------
        samples : array-like
            the samples
        labels : array-like
            the labels
        """
        # TODO: build tree
        pass

    def best_attr_of(self, samples, labels):
        """
        select the best attribute (give max information gain if chosen) from the input samples
        ----------
        samples : array-like
            the sample data
        """
        best_ig = 0.0
        best_attr_idx = None
        for i, attr in enumerate(samples.T):
            curr_ig = self.ig_of(sorted(attr), labels)
            if  curr_ig > best_ig:
                best_ig = curr_ig
                best_attr_idx = i
        best_attr = samples[best_attr_idx, :]
        # TODO: np delete will delete element in the row when there's only one row left
        truncated_samples = np.delete(samples, best_attr_idx, axis=1)
        return best_attr, truncated_samples

    def ig_of(self, attr, labels):
        """
        calculates the information gain if data partitioned by input attr
        ----------
        attr : array-like
            a sorted list of values of a single attribute
        labels : array-like
            a list of values of labels
        """
        # TODO: boolean variable that indicates attr is discrete or continuous
        is_discrete = True
        if is_discrete:
            # attr is discrete
            return self.ig_of_discrete_attr(attr, labels)
        else:
            #TODO: partition cont value
            part_ent = None
            return None

    def ig_of_discrete_attr(self, attr, labels):
        """
        calculates the information gain of input attribute
        """
        og_ent = self.entropy_of(labels)
        attr_label_pair = np.array(attr, labels).T
        unique_symbol = np.unique(attr_label_pair[:, 0])
        best_ig = 0.0
        for symbol in unique_symbol:
            sym = np.array([pair for pair in attr_label_pair if pair[0] == symbol])
            non_sym = np.array([pair for pair in attr_label_pair if pair[0] != symbol])
            # calculate probability of current symbol
            p_sym = len(sym) / float(len(attr_label_pair))
            # calculate entropy of entire attribute using current symbol
            curr_ent = self.entropy_of(sym[:, 1])*p_sym + self.entropy_of(non_sym[:, 1])*(1-p_sym)
            curr_ig = og_ent - curr_ent
            if best_ig < curr_ig:
                best_ig = curr_ig
        return best_ig

    def entropy_of(self, labels):
        """
        calculates entropy of input labels
        ----------
        labels : array-like
            a list of labels
        """
        from collections import Counter
        occurence = list(Counter(labels).values())
        prob = list(map(lambda x: x/float(np.sum(occurence)), occurence))
        entropy = -np.sum(list(map(lambda x: x*np.log2(x), prob)))
        return entropy
