#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Sat Feb 14 17:13:48 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    This script pre-processes (tokenise + normalise) the contents of the
    data/txt_sentoken.all.dat such that each line contains the complete (or
    partial) contents of reviews.

Usage:
    do_norm_poldata.py [-s SEP]


Options:
    -s SEP, --sep SEP   [default: @@SEP@@]


Example workflow:
    ./do_pipeline.py < txt_sentoken.all.dat > txt_sentoken.dat

TODO:
    - user-specified classifiers
    - user-specified classifier parameters.
    - n_trees vs. oob accuracy plot for RF classifier
    - saving/loading of trained models.
    - classifier accuracy/ROC-AUC comparisons

"""

import os
import sys
import pandas as pd
import re

from collections import OrderedDict
from docopt import docopt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from utils import *


def iter_reviews(df, firstn=None, lastn=None):
    """

    Arguments:
        df --
        firstn -- use only the first n sentences of each review.
        lastn -- use only the last n sentences of each review.

    """

    for (label, rev_id), subdf in (df.groupby(["label", "rev_id"])):
        # construct the complete review from the list of sentences.
        complete_review = " ".join(subdf.sentence.tolist())
        yield label, rev_id, complete_review

    return


if __name__ == '__main__':
    opts = docopt(__doc__)

    df = pd.read_table(sys.stdin, sep=opts["--sep"])

    # divide the "id" column into "cv" and "rev_id" columns
    id_split = pd.DataFrame(df.id.str.split("_").tolist(),
                            columns=["cv", "rev_id"])
    df = pd.concat([id_split, df], axis=1)

    stemmer = PorterStemmer()
    records = []

    for label, rev_id, text in iter_reviews(df):
        # tokenise and normalise the sentence text.
        tokens = my_tokenizer(text)
        tokens = [ t for t in tokens if valid_word(t) ]
        tokens = [ t for t in tokens if t not in STOP_WORDS ]

        # reconstruct a "complete" normalised review from the list of tokens.
        line = " ".join(tokens)

        records.append((label, line))

    df = pd.DataFrame.from_records(records, columns=["label", "text"])
    #X, y = df.text, df.label

    df.to_csv(sys.stdout)

    sys.exit()

    #
    # Use the tf-idf statistic to weight the importance of each term.
    #
    from sklearn.cross_validation import cross_val_score
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.ensemble import RandomForestClassifier

    count_vectoriser = CountVectorizer(tokenizer=my_tokenizer)
    tfidf_transformer = TfidfTransformer(use_idf=False)

    X_counts = count_vectoriser.fit(X)
    X_counts = count_vectoriser.transform(X)

    X_tfidf = tfidf_transformer.fit(X_counts)
    X_tfidf = tfidf_transformer.transform(X_counts)

    # Train the model
    clf = RandomForestClassifier(n_estimators=200, oob_score=True, verbose=2, n_jobs=2)
    clf.fit(X_tfidf.toarray(), y)

    #
    # muck around code below!
    #

    # Save (pickle) the model to disk
    from sklearn.externals import joblib

    # Note, sparse input matrices for RF's in sklearn not yet fully supported.
    clf.fit(X_tfidf.toarray(), y)

    joblib.dump(clf, "./mdl/polarity2.0.mdl")

    del clf

    # Load (unpickle) the model
    clf = joblib.load("./mdl/polarity2.0.mdl")

    #
    #print cross_val_score(clf, X_tfidf.toarray(), y)
