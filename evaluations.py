#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Sat Feb 21 13:16:40 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    TODO

Usage:
    evaluations.py heldout  -m <mdl> -c <csv>
    evaluations.py cv       -c <csv> -k <k> [-r <nreps>] [-R <seed>] [-C <classifier>] [<param> ...]

Options:
    -C <classifier>     sklearn classifier name [default: RandomForestClassifier].
    -m <mdl>
    -c <csv>
    -k <nfolds>         Num. of CV folds.
    -r <nreps>          Num. of CV repetitions [default: 1].
    -R <seed>           Integer random seed [default: 54321].  

"""

import os
import sys
import pandas
import sentiment as snt

from docopt import docopt


def do_heldout(opts, ptab_fn):
    """
    """

    clf, vocabulary = snt.load_model(opts["-m"])
    df = pandas.read_csv(opts["-c"], index_col=None, header=None)
    y, X = df[0], df[1]
    X_tfidf, X_vocab = snt.make_tfidf_matrix(X, toarray=True,
            vocabulary=vocabulary)
    y_pred, y_score = clf.predict(X_tfidf), clf.predict_proba(X_tfidf)
    pos_score = pandas.Series(y_score[:,1])

    # compute the ptab dataframe and write it to disk.
    ptab_df = pandas.concat(
            [pandas.Series(y),
             pandas.Series(y_pred),
             pandas.Series(y_score[:,1])], axis=1)
    ptab_df.to_csv(ptab_fn, sep=" ", index=None, header=None)

    return ptab_df


if __name__ == '__main__':
    opts = docopt(__doc__)

    if opts["heldout"]:
        do_heldout(opts)
    elif opts["cv"]:
        raise NotImplementedError
