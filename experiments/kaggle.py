#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Mon Feb 23 18:50:10 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    TODO

Usage:
    kaggle.py (-i <csv>| -D)

Options:
    -i <csv>    Kaggle training/development data.
    -D          Debug/development mode (use small fake data).

"""

import os
import sys
import numpy
import pandas
import sklearn
import sentiment

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification

from sentiment import make_tfidf_matrix
from docopt import docopt

numpy.random.seed(12345)


if __name__ == '__main__':
    opts = docopt(__doc__)

    if opts["-D"]:
        X, y = make_classification(n_features=20, shift=None, scale=None, n_classes=2)
    else:
        df = pandas.read_csv(opts["-i"])
        (X, X_vocab), y = make_tfidf_matrix(df.Phrase), df.Sentiment

    print X.shape
    sys.exit()

    cv = StratifiedKFold(y, n_folds=5)
    cv_scores = cross_val_score(RandomForestClassifier(n_estimators=1000),
            X=X, y=y, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1, pre_dispatch=2)

    print cv_scores
