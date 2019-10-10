# ID3 Decision Tree

This is an implementation of ID3 decision tree algorithm.

## Structure
```ID3/dtree.py``` is the main file to be executed
```ID3/DT_Model.py``` is the model class file that builds a ID3 decision tree

## Usage

```shell
python3 ID3/dtree.py [data_folder_path] use_full_sample max_depth use_gain_ratio
```

where ```use_full_sample``` takes value in {0, 1}, with 0 meaning do a k-fold cross validation (defaulted to 5) and 1 meaning train the model on all datapoints.

```max_depth``` takes value in non-negative integers with 0 representing "grow a full tree".

```use_gain_ratio``` takes value in {0, 1}

# Naive Bayes Net

The first part of `stats_models` is the implementation of the naive Bayes algorithm.

## Structure
```stats_models/nbayes.py``` is the main file to be executed to run naive Bayes algorithm

## Usage

```shell
python3 stats_models/nbayes.py [data_folder_path] use_full_sample max_depth use_gain_ratio
```

```use_full_sample``` takes value in {0, 1}, with 0 meaning do a k-fold cross validation (defaulted to 5) and 1 meaning train the model on all datapoints.

```n_bins``` takes value in non-negative integers greater than 2.

```m_estimate``` takes value in {0, 1}.

# Logistic Regression

The second part of `stats_models` is the implementation of logistic regression algorithm.

## Structure
```stats_models/logreg.py``` is the main file to be executed to run logistic regression.

## Usage

```shell
python3 stats_models/logreg.py [data_folder_path] use_full_sample lambda
```

```use_full_sample``` takes value in {0, 1}, with 0 meaning do a k-fold cross validation (defaulted to 5) and 1 meaning train the model on all datapoints.

```lambda``` is a non-negative real number that sets value for required constant lambda.

# Bagging

The first part of `ensemble_models` is the implementation of bagging algorithm.

## Structure
```ensemble_models/bag.py``` is the main file to be executed to run bagging algorithm

## Usage

```shell
python3 ensemble_models/bag.py [data_folder_path] use_full_sample learning_algo n_iter
```

```use_full_sample``` takes value in {0, 1}, with 0 meaning do a k-fold cross validation (defaulted to 5) and 1 meaning train the model on all datapoints.

```learning_algo``` takes a string among "dtree", "nbayes" and "logreg", representing creating an ensemble of which learning algorithm.

```n_iter``` takes a positive integer, representing the number of iterations to perform bagging.

# Boosting

The second part of `ensemble_models` is the implementation of boosting algorithm.

## Structure
```ensemble_models/boost.py``` is the main file to be executed to run boosting algorithm

## Usage
```shell
python3 ensemble_models/boost.py [data_folder_path] use_full_sample learning_algo n_iter
```

```use_full_sample``` takes value in {0, 1}, with 0 meaning do a k-fold cross validation (defaulted to 5) and 1 meaning train the model on all datapoints.

```learning_algo``` takes a string among "dtree", "nbayes" and "logreg", representing creating an ensemble of which learning algorithm.

```n_iter``` takes a positive integer, representing the number of iterations to perform boosting.