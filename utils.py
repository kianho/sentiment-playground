#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Mon Feb 16 15:40:16 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    ...

"""

import os
import sys
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


RE_SPACE = re.compile(r"\s+")

# Characters that cannot appear within words, taken from nltk.tokenize.punkt.
# Note that some punctuation characters are allowed, e.g. '.' and ','.
RE_NON_WORD_CHARS = re.compile(r"(?:[?!)\";}\]\*:@\'\({\[])")

# Multi-character punctuation, e.g. ellipsis, en-, and em-dashes.
RE_MULTI_CHAR_PUNCT = re.compile(r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)")

# Integer or floating-point number.
RE_NUMBER = re.compile(r"\d+\.?\d*")

# Single non-alphabetic character.
RE_SINGLE_NON_ALPHA = re.compile(r"^[^a-zA-Z]$")

RE_ALPHA = re.compile(r"[a-zA-Z]+")

STOP_WORDS = set(stopwords.words("english"))

PORTER_STEMMER = PorterStemmer()


def valid_word(w):
    """
    """
    return not (
        RE_NON_WORD_CHARS.match(w) or
        RE_MULTI_CHAR_PUNCT.match(w) or
        RE_NUMBER.match(w) or
        RE_SINGLE_NON_ALPHA.match(w)
    ) and RE_ALPHA.match(w)


def remove_non_words(tokens):
    return [ t for t in tokens if valid_word(t) ]


def remove_stop_words(tokens, stop_words=STOP_WORDS):
    return [ t for t in tokens if t not in stop_words ]


def stem_words(tokens):
    return [ PORTER_STEMMER.stem(t) for t in tokens ]


def normalize_tokens(tokens):
    return [ t.lower() for t in remove_non_words(
             remove_stop_words( stem_words(tokens))) ]

def make_tfidf_matrix(documents):
    """TODO
    """

    return
