#!usr/bin/python3
"""
This is an implementation of ID3 decision tree https://en.wikipedia.org/wiki/ID3_algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np
from treelib import Node, Tree
from collections import Counter

REMOVE_ATTRIBUTE = True

class ID3(object):
    """
    a ID3 decision tree
    """

    def __init__(self, max_depth, use_gain_ratio):
        self.max_depth = max_depth
        self.use_gain_ratio = use_gain_ratio
        # ID3
        self.pos_branch = None
        self.neg_branch = None

        self.attr_idx = None
        self.part_val = None

    def fit(self, samples, labels):
        """
        build a id3 decision tree with input samples and labels
        ---------
        samples : array-like
            the samples
        labels : array-like
            the labels
        return : whether or not self is a leaf node
        """
        # base case: max depth reached / pure node / run out of attributes
        if self.max_depth == 0 or self.entropy_of(labels) == 0 or np.size(samples, 1) == 0:
            # create a leaf node with major label, id=-1 indicates leaf node
            self.attr_idx = -1
            self.part_val = self.major_label(labels)
            return

        # recursive case: build subtrees
        self.attr_idx, self.part_val = self.best_attr_of(samples, labels)
        # partition the samples and labels
        pos_subs, neg_subs, pos_labels, neg_labels = self.partition(samples, labels, self.attr_idx, self.part_val)

        # init two branches
        self.pos_branch = ID3(self.max_depth - 1, self.use_gain_ratio)
        self.neg_branch = ID3(self.max_depth - 1, self.use_gain_ratio)

        # recursively build tree
        self.pos_branch.fit(pos_subs, pos_labels)
        self.neg_branch.fit(neg_subs, neg_labels)

    def predict(self, x):
        """
        predict the samples' class labels
        ----------
        samples : array-like
            the sample data
        """
        # TODO: implement predict func
        # self is leaf node
        if self.attr_idx == -1:
            return self.part_val

        attr = x[self.attr_idx]
        if isinstance(attr, float):
            if attr <= self.part_val:
                return self.pos_branch.predict(np.delete(x, self.attr_idx))
            return self.neg_branch.predict(np.delete(x, self.attr_idx))
        if attr == self.part_val:
            return self.pos_branch.predict(np.delete(x, self.attr_idx))
        return self.neg_branch.predict(np.delete(x, self.attr_idx))

    def best_attr_of(self, samples, labels):
        """
        select the best attribute (give max information gain if chosen) from the input samples
        ----------
        samples : array-like
            the sample data
        """
        # TODO: add if use_gr, then use gr_of()
        best_ig = 0.0
        best_attr_idx = None
        for i, attr in enumerate(samples.T):
            curr_ig, curr_partition = self.ig_of(attr, labels)
            if best_ig <= curr_ig:
                best_ig = curr_ig
                best_partition = curr_partition
                best_attr_idx = i
        return best_attr_idx, best_partition

    def ig_of(self, attr, labels):
        """
        calculates the information gain if data partitioned by input attr
        ----------
        attr : array-like
            a sorted list of values of a single attribute
        labels : array-like
            a list of values of labels
        """
        if isinstance(attr[0], float):
            # attr is continuous
            return self.ig_of_cont_attr(attr, labels)
        # attr is discrete or boolean
        return self.ig_of_discrete_attr(attr, labels)

    def ig_of_discrete_attr(self, attr, labels):
        """
        calculates the information gain of input attribute
        :returns : best_ig, best_symbol
        """
        og_ent = self.entropy_of(labels)
        xy_pair = np.array([attr, labels]).T
        unique_symbol = np.unique(xy_pair[:, 0])
        best_ig = 0.0
        for symbol in unique_symbol:
            sym = xy_pair[xy_pair[:, 0] == symbol]
            non_sym = xy_pair[xy_pair[:, 0] != symbol]
            # calculate probability of current symbol
            p_sym = len(sym) / float(len(xy_pair))
            # calculate entropy of entire attribute using current symbol
            curr_ent = self.entropy_of(sym[:, 1])*p_sym + self.entropy_of(non_sym[:, 1])*(1-p_sym)
            curr_ig = og_ent - curr_ent
            if best_ig <= curr_ig:
                best_ig = curr_ig
                best_symbol = symbol
        return best_ig, best_symbol

    def ig_of_cont_attr(self, attr, labels):
        """
        calculates entropy of input continuous labels
        ----------
        labels : array-like
            a list of labels
        :return : best_ig, best_partition
        """
        og_ent = self.entropy_of(labels)
        cont_xy_pair = np.array([attr, labels]).T
        # Sort the attribute label ascendingly
        sorted_xy_pair = cont_xy_pair[cont_xy_pair[:, 0].argsort(kind='mergesort')]
        # list of the indexes of samples where class label changed
        changed_idx = np.where(sorted_xy_pair[:, 1] != np.roll(sorted_xy_pair[:, 1], 1))[0]
        best_ig = -100.0
        for i in changed_idx[1:]:
            # partition list into 2 branches
            curr_partition = (sorted_xy_pair[:, 0][i] + sorted_xy_pair[:, 0][i-1])/2
            left_branch = sorted_xy_pair[sorted_xy_pair[:, 0] <= curr_partition]
            right_branch = sorted_xy_pair[sorted_xy_pair[:, 0] > curr_partition]
            # calculate information gain of current split
            p_left = (left_branch.shape[0])/float(sorted_xy_pair.shape[0])
            p_right = (right_branch.shape[0])/float(sorted_xy_pair.shape[0])
            curr_ent = p_left*self.entropy_of(left_branch[:, 1]) + p_right*self.entropy_of(right_branch[:, 1])
            curr_ig = og_ent - curr_ent
            # Finding the maximum information gain
            if best_ig <= curr_ig:
                best_ig = curr_ig
                best_partition = curr_partition
        return best_ig, best_partition

    def entropy_of(self, labels):
        """
        calculates entropy of input labels
        ----------
        labels : array-like
            a list of labels
        """
        occurence = list(Counter(labels).values())
        prob = list(map(lambda x: x/float(np.sum(occurence)), occurence))
        entropy = -np.sum(list(map(lambda x: x*np.log2(x), prob)))
        return entropy

    def gr_of(self, attr, labels):
        """
        calculates the gain ratio of input attribute
        ----------
        attr : array-like
            a sorted list of values of a single attribute
        labels : array-like
            a list of values of labels
        """
        information_gain = self.ig_of(attr, labels)
        entropy = self.entropy_of(attr)
        gain_ratio = information_gain / float(entropy)
        return gain_ratio

    def partition(self, samples, labels, attr_idx, part_value):
        """
        partitions the samples and labels by input attribute and partition value
        ----------
        samples : array-like
            the sample data
        labels : array-like
            the label data
        attr_idx: int
            the index of the selected attribute taken as node
        part_value : float or String
            the mid value we partition data with
        """

        # get the indexs of samples which are positive according to the partition
        if isinstance(samples[0, attr_idx], float):
            index = np.where(samples[:, attr_idx] <= part_value)[0]
        else:
            index = np.where(samples[:, attr_idx] == part_value)[0]

        # get the to subset of samples by positiveness and negativeness
        pos_subs = samples[index]
        neg_subs = np.delete(samples, index, axis=0)
        pos_labels = labels[index]
        neg_labels = np.delete(labels, index, axis=0)
        # remove attribute
        if REMOVE_ATTRIBUTE:
            pos_subs = np.delete(pos_subs, attr_idx, axis=1)
            neg_subs = np.delete(neg_subs, attr_idx, axis=1)

        return pos_subs, neg_subs, pos_labels, neg_labels

    def major_label(self, labels):
        keys = list(Counter(labels).keys())
        # if only contain one class
        if len(keys) == 1:
            return keys[0]
        # if contains nothing (anomaly
        elif not keys:
            raise Exception("EMPTY_LABELS_ERROR")
        counts = list(Counter(labels).values())
        if counts[0] > counts[1]:
            return keys[0]
        return keys[1]
