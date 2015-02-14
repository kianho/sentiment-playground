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
    do_preprocessing.py [-s SEP]

Options:
    -s SEP, --sep SEP   [default: @@SEP@@]

"""

import os
import sys
import pandas as pd
import re

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from docopt import docopt

RE_SPACE = re.compile(r"\s+")

# Characters that cannot appear within words, taken from nltk.tokenize.punkt.
# Note that some punctuation characters are allowed, e.g. '.' and ','.
RE_NON_WORD_CHARS = re.compile(r"(?:[?!)\";}\]\*:@\'\({\[])")

# Multi-character punctuation, e.g. ellipsis, en-, and em-dashes.
RE_MULTI_CHAR_PUNCT = re.compile(r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)")

# Integer or floating-point number.
RE_NUMBER = re.compile(r"^\d+\.?\d*$")

# Single non-alphabetic character.
RE_SINGLE_NON_ALPHA = re.compile(r"^[^a-zA-Z]$")

STOP_WORDS = set(stopwords.words("english"))


def valid_word(w):
    return not (
        RE_NON_WORD_CHARS.match(w) or
        RE_MULTI_CHAR_PUNCT.match(w) or
        RE_NUMBER.match(w) or
        RE_SINGLE_NON_ALPHA.match(w)
    )


def my_tokenizer(s):
    """
    """
    return [ t for t in RE_SPACE.split(s) ]


def iter_reviews(df, firstn=None, lastn=None):
    """
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

    for label, rev_id, text in iter_reviews(df):
        # tokenise and normalise the sentence text.
        tokens = my_tokenizer(text)
        tokens = [ t for t in tokens if valid_word(t) ]
        tokens = [ t for t in tokens if t not in STOP_WORDS ]

        # re-construct the sentence from the normalised tokens.
        line = " ".join(tokens)
        sys.stdout.write(opts["--sep"].join(("%d" % label, rev_id, line))
                            + os.linesep)
