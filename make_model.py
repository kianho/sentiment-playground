#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Sun Feb 15 13:37:42 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    ...

Usage:
    make_model.py -d DAT -o MDL

Options:
    -d DAT  Read training data from a .dat file.
    -o MDL  The file in which to save the model.

"""

import os
import sys
import pandas as pd

from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from docopt import docopt
from utils import RE_SPACE


if __name__ == '__main__':
    opts = docopt(__doc__)

    df = pd.read_csv(opts["-d"])

    X, Y = df.text, df.label

    #
    # Use the tf-idf statistic to weight the importance of each term.
    #
    count_vectoriser = CountVectorizer(tokenizer=lambda s : RE_SPACE.split(s))
    tfidf_transformer = TfidfTransformer(use_idf=False)

    X_counts = count_vectoriser.fit(X)
    X_counts = count_vectoriser.transform(X)

    X_tfidf = tfidf_transformer.fit(X_counts)
    X_tfidf = tfidf_transformer.transform(X_counts)

    # Train the model
    clf = RandomForestClassifier(n_estimators=200, oob_score=True, verbose=2, n_jobs=2)
    clf.fit(X_tfidf.toarray(), Y)

    # Save (pickle) the model to disk
    joblib.dump(clf, opts["-o"])
