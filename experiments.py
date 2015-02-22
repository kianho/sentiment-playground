#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Fri Feb 20 17:21:41 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    TODO

Usage:
    example.py

Options:

"""

import os
import sys
import pprint
import numpy
import pandas
import sklearn
import sentiment

from sentiment import tokenize
from sentiment import make_tfidf_matrix, read_vocab, PORTER_STEMMER, STOP_WORDS
from sentiment import RE_PUNCT_WORD, RE_NON_WORD, RE_RATING, RE_NUMBER_ONLY
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import make_classification
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.cross_validation import StratifiedKFold

from docopt import docopt

# Initialise the global numpy random seed.
numpy.random.seed(12345)

# Dataset
csv_fn = "./data/polarity2.0/polarity2.0.csv"

# Load the raw dataset.
df = pandas.read_csv(csv_fn, index_col=None, header=None)

# UNCOMMENT WHEN READY
# Pre-process the dataset into a sparse tf-idf matrix and vocab. dictionary.
#X, X_vocab = sentiment.make_tfidf_matrix(df[1], toarray=True)
#y = df[0]
X, y = make_classification(n_samples=2000, n_features=50, scale=None, shift=None)

# Scale to unit variance with a 0 mean.
X = sklearn.preprocessing.scale(X)

# Cross-validation generator and instantiate the folds in a list so we can
# re-use them for other estimators.
cv_gen =  StratifiedKFold(y, n_folds=10, shuffle=True)
cv_folds = list(cv_gen)

# Perform a series of cross-validation grid searches over several different
# types of classifiers.

# Grid search over the SGDClassifier loss functions.
sgdc_grid =  { "loss" : ["hinge", "log"] }
sgdc_grid_clf = GridSearchCV(SGDClassifier(random_state=12345), param_grid=sgdc_grid, cv=cv_folds,
        scoring="roc_auc", verbose=3, n_jobs=-1)
sgdc_grid_clf.fit(X, y)

# Grid search over naive Bayes (empty grid i.e. standard 10-fold cv).
nbayes_clf = GridSearchCV(GaussianNB(), param_grid={}, cv=cv_folds,
        scoring="roc_auc", verbose=3, n_jobs=-1)
nbayes_clf.fit(X, y)

# Random forest.
_, n_feats = X.shape

rf_grid_clf = GridSearchCV(RandomForestClassifier(n_estimators=1000),
        param_grid={}, cv=cv_folds, scoring="roc_auc", verbose=3, n_jobs=-1)

# Extremely random trees.
ert_grid_clf = GridSearchCV(ExtraTreesClassifier(n_estimators=1000),
        param_grid={}, cv=cv_folds, scoring="roc_auc", verbose=3, n_jobs=-1)

# SVMs
C = [.1, 1, 10, 100]
svm_grid = [
        { "kernel" : ["linear"], "C" : C },
        { "kernel" : ["rbf"], "gamma" : [0.1, 0.2, 0.5, 1.0], "C" : C },
        { "kernel" : ["poly"], "degree" : [2,3], "C" : C }
    ]
svm_grid_clf = GridSearchCV(SVC(), param_grid=svm_grid, scoring="roc_auc",
        verbose=3, n_jobs=-1)
print svm_grid_clf.fit(X, y).grid_scores_
