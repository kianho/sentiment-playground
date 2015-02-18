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

from functools import reduce

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


RE_SPACE = re.compile(r"\s+")

# Characters that cannot appear within words, taken from nltk.tokenize.punkt.
# Note that some punctuation characters are allowed, e.g. '.' and ','.
NON_WORD_CHARS_PAT = r"(?:[?!)\";}\]\*:@\'\({\[])"
RE_NON_WORD_CHARS = re.compile(NON_WORD_CHARS_PAT)

# Multi-character punctuation, e.g. ellipsis, en-, and em-dashes.
# TODO:
# - more accurate way of removing punctuations
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
    TODO:
    - use single regular expression composed of individual patterns.
    """
    return not (
        RE_NON_WORD_CHARS.match(w) or
        RE_MULTI_CHAR_PUNCT.match(w) or
        RE_NUMBER.match(w) or
        RE_SINGLE_NON_ALPHA.match(w)
    ) and RE_ALPHA.match(w)


# TODO: 
# - convert these functions to return iterators

def remove_non_words(tokens):
    return [ t for t in tokens if valid_word(t) ]


def remove_stop_words(tokens, stop_words=STOP_WORDS):
    return [ t for t in tokens if t not in stop_words ]


def stem_words(tokens):
    return [ PORTER_STEMMER.stem(t) for t in tokens ]


def normalize_tokens(tokens):
    return [ t.lower() for t in remove_non_words(
             remove_stop_words( stem_words(tokens))) ]


def preprocess_blob(blob, remove_html=False):
    """Preprocess raw text (e.g. from a text file, html page, twitter stream
    etc.) by normalising and tokenizing into a list of more usable document
    terms.

    Arguments:
        blob --
        remove_html --

    Returns:
        a single list of normalised terms associated with a single document
        blob.

    """

    tokens = reduce(lambda x, y : x + y,
                map(word_tokenize, sent_tokenize(blob.lower())))
    terms = normalize_tokens(tokens)

    return terms


def make_tfidf_matrix(documents):
    """TODO
    """

    return


if __name__ == "__main__":
    blob = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
    print preprocess_blob(blob)
