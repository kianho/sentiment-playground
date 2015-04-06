#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Mon Feb 23 22:38:29 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    ...

Usage:
    features.py extract -i <csv> -o <svml>
    features.py select -i <csv> -o <svml>

Options:
    -i <csv>
    -o <svml>

"""

import os
import sys
import pandas

from docopt import docopt
from sentiment import make_tfidf_matrix

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC

def do_select(X, y):
    selector = SelectKBest(chi2, k=2000)
    X = selector.fit_transform(X, y)
    return X


if __name__ == '__main__':
    opts = docopt(__doc__)

    if opts["extract"]:
        df = pandas.read_csv(opts["-i"])
        (X, _), y = make_tfidf_matrix(df.Phrase), df.Sentiment
        dump_svmlight_file(X, y, opts["-o"])
    elif opts["select"]:
        X, y = load_svmlight_file(opts["-i"])
        X = do_select(X, y)

        # TODO: add a classifier here

    assert(os.path.exists(opts["-o"]))
