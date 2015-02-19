#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Wed Feb 18 23:03:26 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    ...

Usage:
    make_sentiment.py -m MDL -V DAT (-t TXT | --)

Options:
    -m MDL
    -V DAT
    -t TXT  Read text from a file.
    --      Read text from stdin

"""

import os
import sys
import pandas

from sklearn.externals import joblib
from utils import preprocess_blob, make_tfidf_matrix
from docopt import docopt


if __name__ == '__main__':
    opts = docopt(__doc__)

    vocab = dict(pandas.read_csv(opts["-V"]).to_records(index=None))

    with (sys.stdin if opts["--"] else open(opts["-t"])) as f:
        blob = f.read()

    X, _ = make_tfidf_matrix([preprocess_blob(blob)], vocab=vocab)

    clf = joblib.load(opts["-m"])

    print clf.predict(X.toarray())
