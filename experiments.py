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
import numpy
import pandas

from sentiment import tokenize
from sentiment import make_tfidf_matrix, read_vocab, PORTER_STEMMER, STOP_WORDS
from sentiment import RE_PUNCT_WORD, RE_NON_WORD, RE_RATING, RE_NUMBER_ONLY
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from docopt import docopt


if __name__ == '__main__':
    csv_fn = "./data/polarity2.0/polarity2.0.csv"
    mdl_fn = "./mdl/polarity2.0.example.mdl"
    vocab_fn = "./mdl/polarity2.0.example.vocab.csv"

    if os.path.exists(mdl_fn):
        clf = joblib.load(mdl_fn)
        vocab = dict((v,k) for k,v in read_vocab(vocab_fn).items())
        var_imps = sorted(((imp, vocab[i]) for (i, imp) in enumerate(clf.feature_importances_)))
        print var_imps
    else:
        df = pandas.read_csv(csv_fn, index_col=None, header=None)
        y, X = df[0], df[1]

        X_tfidf, X_vocab = make_tfidf_matrix(X,
                tokenizer=lambda s : tokenize(s, stop_words=STOP_WORDS))

        clf = RandomForestClassifier(n_estimators=500, compute_importances=True,
                oob_score=True, n_jobs=4, verbose=1)
        clf.fit(X_tfidf.toarray(), y)

        # Write the model to disk.
        joblib.dump(clf, mdl_fn)

        # Write the vocabulary to disk
        vocab = pandas.DataFrame(X_vocab.items())
        vocab.to_csv(vocab_fn, index=None, header=None)
