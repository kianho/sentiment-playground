#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Sun Feb 15 13:37:42 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    ...

    Important--this script assumes that all the training data and text has been
    preprocessed and normalised. The only text manipulation required is to
    obtain the individual document terms by splitting each string using
    str.split().

Usage:
    make_model.py -d DAT -m MDL -V DAT [-c CLASSIFIER] 

Options:
    -d DAT  Read training data from a .dat file.
    -m MDL  The file in which to save the model.
    -V DAT  The vocabulary used in the model.
    -c CLASSIFIER

"""

import os
import sys
import pandas

from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from docopt import docopt
from utils import normalize_tokens, make_tfidf_matrix

DEFAULT_CLF_OPTS = \
        "RandomForestClassifier,n_estimators=200,oob_score=True,n_jobs=1"

if __name__ == '__main__':
    opts = docopt(__doc__)

    #DEFAULT_CLF_OPTS.split(",")

    df = pandas.read_csv(opts["-d"])
    df.text = df.text.apply(lambda s : s.split())
    X, Y = df.text, df.label

    #
    # Use the tf-idf statistic to weight the importance of each term.
    #
    X_tfidf, X_vocab = make_tfidf_matrix(X.tolist())

    # Train the model
    clf = RandomForestClassifier(n_estimators=10, oob_score=True, verbose=2, n_jobs=2)
    clf.fit(X_tfidf.toarray(), Y)

    # Save the vocabulary to disk
    vocab = pandas.DataFrame(X_vocab.items(), columns=["term", "index"])
    vocab.to_csv(opts["-V"], index=None)

    # Save (pickle) the model to disk
    joblib.dump(clf, opts["-m"])
