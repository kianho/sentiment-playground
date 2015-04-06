#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Mon Feb 23 19:03:14 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    TODO

Usage:
    make_dev_csv.py -i <csv> -o <csv>

Options:
    -i <csv>
    -o <csv>

"""

import os
import sys
import numpy
import sklearn
import pandas

from sklearn.cross_validation import StratifiedShuffleSplit
from docopt import docopt

numpy.random.seed(12345)


if __name__ == '__main__':
    opts = docopt(__doc__)

    df = pandas.read_csv(opts["-i"])
    cv_iter = StratifiedShuffleSplit(df.Sentiment, n_iter=1, train_size=0.33)
    train_ix, _ = list(cv_iter)[0]
    df.iloc[train_ix].to_csv(opts["-o"], index=False)
