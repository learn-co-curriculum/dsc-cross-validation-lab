
# Introduction to Cross-Validation - Lab

## Introduction

In this lab, you'll be able to practice your cross-validation skills!


## Objectives

You will be able to:

- Perform cross validation on a model to determine optimal model performance
- Compare training and testing errors to determine if model is over or underfitting

## Let's get started

This time, let's only include the variables that were previously selected using recursive feature elimination. We included the code to pre-process below.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_boston

boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
b = boston_features['B']
logdis = np.log(boston_features['DIS'])
loglstat = np.log(boston_features['LSTAT'])

# Min-Max scaling
boston_features['B'] = (b-min(b))/(max(b)-min(b))
boston_features['DIS'] = (logdis-min(logdis))/(max(logdis)-min(logdis))

# Standardization
boston_features['LSTAT'] = (loglstat-np.mean(loglstat))/np.sqrt(np.var(loglstat))
```


```python
X = boston_features[['CHAS', 'RM', 'DIS', 'B', 'LSTAT']]
y = pd.DataFrame(boston.target, columns = ['target'])
type(X)
```

### Train-test split

Perform a train-test split with a test set of 20%.


```python
# Import train_test_split from sklearn.model_selection

```


```python
# Split the data into training and test sets (assign 20% to test set)

```


```python
# A brief preview of train-test split
print(len(X_train), len(X_test), len(y_train), len(y_test))
```

### Fit the model

Fit a linear regression model and apply the model to make predictions on test set


```python

```

### Residuals and MSE

Calculate the residuals and the mean squared error on the test set


```python

```

## Cross-Validation: let's build it from scratch!

### Create a cross-validation function

Write a function `kfolds()` that splits a dataset into k evenly sized pieces. If the full dataset is not divisible by k, make the first few folds one larger then later ones.

We want the folds to be a list of subsets of data!


```python
def kfolds(data, k):
    # Force data as pandas DataFrame
    # add 1 to fold size to account for leftovers           
    return None
```

### Apply it to the Boston Housing data


```python
# Make sure to concatenate the data again
bos_data = None
```


```python
# Apply kfolds() to bos_data with 5 folds

```

### Perform a linear regression for each fold and calculate the training and test error

Perform linear regression on each and calculate the training and test error: 


```python
test_errs = []
train_errs = []
k=5

for n in range(k):
    # Split in train and test for the fold
    train = None
    test = None
    # Fit a linear regression model
    
    # Evaluate Train and Test errors

# print(train_errs)
# print(test_errs)
```

## Cross-Validation using Scikit-Learn

This was a bit of work! Now, let's perform 5-fold cross-validation to get the mean squared error through scikit-learn. Let's have a look at the five individual MSEs and explain what's going on.


```python

```

Next, calculate the mean of the MSE over the 5 cross-validation and compare and contrast with the result from the train-test split case.


```python

```

##  Summary 

Congratulations! You are now familiar with cross-validation and know how to use `cross_val_score()`. Remember that the results obtained from cross-validation are robust and always use it whenever possible! 
