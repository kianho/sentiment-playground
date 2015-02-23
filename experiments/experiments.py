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


class MyGridSearchCV:
    """
    """

    def __init__(self, param_grids, **kwargs):
        """Constructor.

        """

        # Cross-validation generator and instantiate the folds in a list so we can
        # re-use them for other estimators.
        kwargs.setdefault("cv",
                list(StratifiedKFold(y, n_folds=10, shuffle=True)))
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("scoring", "roc_auc")

        self.param_grids_ = param_grids
        self.grid_search_cv_kwargs_ = kwargs
        self.grid_scores_ = []

        return

    def fit(self, X, y):
        """
        """

        kwargs = self.grid_search_cv_kwargs_
        class_name = lambda clf : clf.__class__.__name__

        param_scores = []
        for clf, param_grid in self.param_grids_:
            cv_clf = GridSearchCV(clf, param_grid=param_grid, **kwargs)
            cv_clf.fit(X, y)

            param_scores.extend([
                    { "fold_scores" : x.cv_validation_scores,
                      "params" : x.parameters,
                      "classifier" : clf.__class__.__name__ }
                    for x in cv_clf.grid_scores_
                ])

        # Create a dataframe of grid scores for each possible
        # classifier/parameter combination.
        df = pandas.DataFrame(param_scores)
        df["mean"] = df["fold_scores"].apply(lambda x : x.mean())

        self.grid_scores_ = df

        return




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


# Perform a series of cross-validation grid searches over several different
# types of classifiers.

C = [.1, 1, 10, 100]
svm_grid = [
        { "kernel" : ["linear"], "C" : C },
        { "kernel" : ["rbf"], "gamma" : [0.1, 0.2, 0.5, 1.0], "C" : C },
        { "kernel" : ["poly"], "degree" : [2,3], "C" : C }
    ]

grid_search_cv = MyGridSearchCV([
        (SGDClassifier(), { "loss" : ["hinge", "log"] }),
        (GaussianNB(), {}),
        (RandomForestClassifier(n_estimators=1000), {}),
        (ExtraTreesClassifier(n_estimators=1000), {}),
        (SVC(), svm_grid)
    ], verbose=0)

grid_search_cv.fit(X, y)
