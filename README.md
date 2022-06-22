# Introduction to Cross-Validation - Lab

## Introduction

In this lab, you'll be able to practice your cross-validation skills!


## Objectives

You will be able to:

- Perform cross validation on a model
- Compare and contrast model validation strategies

## Let's Get Started

We included the code to pre-process the Ames Housing dataset below. This is done for the sake of expediency, although it may result in data leakage and therefore overly optimistic model metrics.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

ames = pd.read_csv('ames.csv')

continuous = ['LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
categoricals = ['BldgType', 'KitchenQual', 'SaleType', 'MSZoning', 'Street', 'Neighborhood']

ames_cont = ames[continuous]

# log features
log_names = [f'{column}_log' for column in ames_cont.columns]

ames_log = np.log(ames_cont)
ames_log.columns = log_names

# normalize (subract mean and divide by std)

def normalize(feature):
    return (feature - feature.mean()) / feature.std()

ames_log_norm = ames_log.apply(normalize)

# one hot encode categoricals
ames_ohe = pd.get_dummies(ames[categoricals], prefix=categoricals, drop_first=True)

preprocessed = pd.concat([ames_log_norm, ames_ohe], axis=1)

X = preprocessed.drop('SalePrice_log', axis=1)
y = preprocessed['SalePrice_log']
```

## Train-Test Split

Perform a train-test split with a test set of 20% and a random state of 4.


```python
# Import train_test_split from sklearn.model_selection

```


```python
# Split the data into training and test sets (assign 20% to test set)

```

### Fit a Model

Fit a linear regression model on the training set


```python
# Import LinearRegression from sklearn.linear_model

```


```python
# Instantiate and fit a linear regression model

```

### Calculate MSE

Calculate the mean squared error on the test set


```python
# Import mean_squared_error from sklearn.metrics

```


```python
# Calculate MSE on test set

```

## Cross-Validation using Scikit-Learn

Now let's compare that single test MSE to a cross-validated test MSE.


```python
# Import cross_val_score from sklearn.model_selection

```


```python
# Find MSE scores for a 5-fold cross-validation

```


```python
# Get the average MSE score

```

Compare and contrast the results. What is the difference between the train-test split and cross-validation results? Do you "trust" one more than the other?


```python
# Your answer here

```

## Level Up: Let's Build It from Scratch!

### Create a Cross-Validation Function

Write a function `kfolds(data, k)` that splits a dataset into `k` evenly sized pieces. If the full dataset is not divisible by `k`, make the first few folds one larger then later ones.

For example, if you had this dataset:


```python
example_data = pd.DataFrame({
    "color": ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
})
example_data
```

`kfolds(example_data, 3)` should return:

* a dataframe with `red`, `orange`, `yellow`
* a dataframe with `green`, `blue`
* a dataframe with `indigo`, `violet`

Because the example dataframe has 7 records, which is not evenly divisible by 3, so the "leftover" 1 record extends the length of the first dataframe.


```python
def kfolds(data, k):
    folds = []
    
    # Your code here
    
    return folds
```


```python
results = kfolds(example_data, 3)
for result in results:
    print(result, "\n")
```

### Apply Your Function to the Ames Housing Data

Get folds for both `X` and `y`.


```python
# Apply kfolds() to ames_data with 5 folds

```

### Perform a Linear Regression for Each Fold and Calculate the Test Error

Remember that for each fold you will need to concatenate all but one of the folds to represent the training data, while the one remaining fold represents the test data.


```python
# Replace None with appropriate code
test_errs = []
k = 5

for n in range(k):
    # Split into train and test for the fold
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    
    # Fit a linear regression model
    None
    
    # Evaluate test errors
    None

print(test_errs)
```

If your code was written correctly, these should be the same errors as scikit-learn produced with `cross_val_score` (within rounding error). Test this out below:


```python
# Compare your results with sklearn results

```

This was a bit of work! Hopefully you have a clearer understanding of the underlying logic for cross-validation if you attempted this exercise.

##  Summary 

Congratulations! You are now familiar with cross-validation and know how to use `cross_val_score()`. Remember that the results obtained from cross-validation are more robust than train-test split.
