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
import numpy

from functools import reduce

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
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
        remove_html -- TODO, 

    Returns:
        a single list of normalised terms associated with a single document
        blob.

    """

    tokens = reduce(lambda x, y : x + y,
                map(word_tokenize, sent_tokenize(blob.lower())))
    terms = normalize_tokens(tokens)

    return terms


def make_tfidf_matrix(doc_term_lists, vocab=None):
    """

    Arguments:
        doc_term_lists:
            list of document term lists.

        vocab (default: None):
            ...

    Returns:
        ...

    """

    X = numpy.array([ " ".join(terms) for terms in doc_term_lists ])
    count_vectoriser = CountVectorizer(tokenizer=lambda s : s.split(), vocabulary=vocab)
    tfidf_transformer = TfidfTransformer(use_idf=True)

    X_counts = count_vectoriser.fit(X)
    X_counts = count_vectoriser.transform(X)

    X_tfidf = tfidf_transformer.fit(X_counts)
    X_tfidf = tfidf_transformer.transform(X_counts)

    return X_tfidf, count_vectoriser.vocabulary_


def train_model(clf, dat_fn, mdl_fn, vocab_fn):
    """

    Arguments:
        clf --
        dat_fn --
        mdl_fn --
        vocab_fn --

    Returns:
        ...

    """

    df = pandas.read_csv(dat_fn)
    df.text = df.text.apply(lambda s : s.split())
    X, Y = df.text, df.label

    # Use the standard tf-idf matrix representation of the dataset.
    X_tfidf, X_vocab = make_tfidf_matrix(X.tolist())

    # Train the model
    clf.fit(X_tfidf.toarray(), Y)

    # Save the vocabulary to disk
    vocab = pandas.DataFrame(X_vocab.items(), columns=["term", "index"])
    vocab.to_csv(vocab_fn, index=None)

    # Save (pickle) the model to disk
    joblib.dump(clf, mdl_fn)

    return


if __name__ == "__main__":
    #blob = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
    #print preprocess_blob(blob)
    #print make_tfidf_matrix([ preprocess_blob(b) for b in (blob, blob) ])
    dat_fn = 
    clf = RandomForestClassifier()
    train_model(clf, )

    #print clf
