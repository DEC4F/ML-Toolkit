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
            return self.ig_of_cont_attr(attr, labels)
            #return None

    def ig_of_discrete_attr(self, attr, labels):
        """
        calculates the information gain of input attribute
        """
        og_ent = self.entropy_of(labels)
        attr_label_pair = np.array(attr, labels).T
        unique_symbol = np.unique(attr_label_pair[:, 0])
        best_ig = 0.0
        for symbol in unique_symbol:
            sym = attr_label_pair[attr_label_pair[:, 0] == symbol]
            non_sym = attr_label_pair[attr_label_pair[:, 0] != symbol]
            # calculate probability of current symbol
            p_sym = len(sym) / float(len(attr_label_pair))
            # calculate entropy of entire attribute using current symbol
            curr_ent = self.entropy_of(sym[:, 1])*p_sym + self.entropy_of(non_sym[:, 1])*(1-p_sym)
            curr_ig = og_ent - curr_ent
            if best_ig < curr_ig:
                best_ig = curr_ig
        return best_ig

    def ig_of_cont_attr(self, attr, labels):
        """
        calculates entropy of input continuous labels
        ----------
        labels : array-like
            a list of labels
        """
        og_ent = self.entropy_of(labels)
        cont_attr_label_pair = np.array(attr, labels).T
        # -2 : I took an example where second last column is for the continuous attribute
        sortted_attr_label_pair = cont_attr_label_pair[cont_attr_label_pair[:,0].argsort(kind='mergesort')]
        # Sort the continuous attribute label based in the ascending order
        chng_in_val = np.where(np.roll(sortted_attr_label_pair[:,1],1)!=sortted_attr_label_pair[:,1])[0]
        # chng_in_val :  list of the indexes of the continious attributes where there's a change in the Class Label
        best_ig = 0.0
        # traverse through all the indexes of the continious attributes where there's a change in the Class Label
        for i in chng_in_val[1:]:
            # attribute list with attribute less than or equal-to the found index-of-change
            attr_list_with_TRUE = sortted_attr_label_pair[sortted_attr_label_pair[:,0]<= ((sortted_attr_label_pair[:,0][i] + sortted_attr_label_pair[:,0][i-1])/2)]

            # attribute list with attribute greater than the found index-of-change
            attr_list_with_False = sortted_attr_label_pair[sortted_attr_label_pair[:,0] > ((sortted_attr_label_pair[:,0][i] + sortted_attr_label_pair[:,0][i-1])/2)]

            # probability of attribute list with attribute less than or equal-to the found index-of-change
            probab_with_TRUE = (attr_list_with_TRUE.shape[0])/float(sortted_attr_label_pair.shape[0])

            # probability of attribute list with attribute greater than the found index-of-change
            probab_with_False = (attr_list_with_False.shape[0])/float(sortted_attr_label_pair.shape[0])

            # As H(Y|X) = P(X=TRUE)H(Y|X=TRUE) + P(X=FALSE)H(Y|X=FALSE)
            curr_et = probab_with_TRUE * self.entropy_of_cont(attr_list_with_TRUE) + probab_with_False * self.entropy_of_cont(attr_list_with_False)

            #IG = H(Y) - H(Y|X)
            ig_cont = og_ent - curr_et

            # Finding the maximum information gain
            if best_ig < ig_cont:
                best_ig = ig_cont
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

    def entropy_of_cont(self, labels):
        """
        calculates entropy of input labels
        ----------
        labels : array-like
            a list of labels
        """
        from collections import Counter
        occurence = list(Counter(labels[:, 0]).values())
        prob = [x/float(np.sum(occurence)) for x in occurence]
        boolarr = np.array(labels[:, 1], dtype=np.bool)
        if np.sum(boolarr) == 0:
            entropy = 0.0
        else:
            entropy = -np.sum([x*np.log2(np.sum(boolarr)/float(np.sum(occurence))) for x in prob])
            return entropy
