#!usr/bin/python3
"""
This is an implementation of logistic regression algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""
import sys

def main():
    """
    run the logistic regression with given command line input
    ----------
    """
    file_path, use_full_sample, lbd = sys.argv[1:4]
    # parse args
    [use_full_sample, lbd] = [int(use_full_sample), int(lbd)]
    clf = LogisticRegression(0.01, 10000, 1)
    clf.fit(samples, labels)

if __name__ == '__main__':
    main()
