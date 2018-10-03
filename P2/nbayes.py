#!usr/bin/python3
"""
This is an implementation of naive Bayes algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""
import sys

def main():
    """
    run the naive bayes network with given command line input
    ----------
    """
    file_path, use_full_sample, n_bins, m_estimate = sys.argv[1:5]
    # parse args
    [use_full_sample, n_bins, m_estimate] = [int(use_full_sample), int(n_bins), int(m_estimate)]

if __name__ == '__main__':
    main()
