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
import sklearn

from docopt import docopt
from sklearn.metrics import roc_curve


def do_roc_curve(y_correct, y_score):
    """
    """

    import mpl_preamble
    from mpl_preamble import pylab

    fpr, tpr, thresh = roc_curve(y_correct, y_score, pos_label=1)

    pylab.plot(x=fpr, y=tpr)
    pylab.savefig("./roc.pdf")

    return


def calc_metrics(ptab):
    """
    """

    correct, pred, scores = ptab[0], ptab[1], ptab[2]

    # Compute the confusion matrix
    confusion_mat = \
        sklearn.metrics.confusion_matrix(correct, pred)

    TP, FP = confusion_mat[1][1], confusion_mat[0][1]
    TN, FN = confusion_mat[0][0], confusion_mat[1][0]

    TPR = float(TP) / (TP + FN)
    FPR = float(FP) / (FP + TN)
    TNR = float(TN) / (FP + TN)
    FNR = float(FN) / (FN + TP)

    MCC = sklearn.metrics.matthews_corrcoef(correct, pred)
    AUC = sklearn.metrics.roc_auc_score(correct, scores)

    rec = float(TP) / (TP + FN)
    prec = float(TP) / (TP + FP)

    precision, recall, F1, sup = \
        sklearn.metrics.precision_recall_fscore_support(correct, pred)

    precision_neg, precision_pos = precision
    recall_neg, recall_pos = recall

    # accuracy
    acc = sklearn.metrics.accuracy_score(correct, pred)

    # error-rate
    err = 1 - acc

    # use the ptab file name as the mtab index
    metrics = pandas.DataFrame([{}])

    metrics["auc"] = AUC
    metrics["mcc"] = MCC
    metrics["precision"] = precision_pos
    metrics["recall"] = recall_pos
    metrics["specificity"] = TNR
    metrics["tpr"] = TPR
    metrics["fpr"] = FPR
    metrics["tnr"] = TNR
    metrics["fnr"] = FNR
    metrics["err"] = err
    metrics["acc"] = acc

    return metrics


def do_heldout(opts, ptab_fn):
    """
    """

    clf, vocabulary = snt.load_model(opts["-m"])
    df = pandas.read_csv(opts["-c"], index_col=None, header=None)
    y, X = df[0], df[1]
    X_tfidf, X_vocab = snt.make_tfidf_matrix(X, toarray=True,
            vocabulary=vocabulary)
    y_pred, y_score = clf.predict(X_tfidf), clf.predict_proba(X_tfidf)

    # compute the ptab and write it to disk.
    ptab = pandas.concat(
            [pandas.Series(y),
             pandas.Series(y_pred),
             pandas.Series(y_score[:,1])], axis=1)
    ptab.to_csv(ptab_fn, sep=" ", index=None, header=None)

    return ptab


def do_cv(opts):
    """
    """
    df = pandas.read_csv(opts["-c"], index_col=None, header=None)
    y, X = df[0], df[1]
    X_tfidf, X_vocab = snt.make_tfidf_matrix(X, toarray=True)

    from sklearn import cross_validation
    from sklearn import svm

    clf = svm.SVC()

    print cross_validation.cross_val_score(clf, X_tfidf, y, n_jobs=-1)

    return


if __name__ == '__main__':
    opts = docopt(__doc__)

    if opts["heldout"]:
        ptab = do_heldout(opts, "/dev/null")

        # TODO: annotate each result row with classifier parameters and dataset
        # name.
        mtab = calc_metrics(ptab)
    elif opts["cv"]:
        do_cv(opts)
