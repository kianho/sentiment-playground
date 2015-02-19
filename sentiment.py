#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Mon Feb 16 15:40:16 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    ...

Usage:
    sentiment.py train -c <csv> [-C <str>] -m <mdl> -V <csv> [-e <encoding>]
    sentiment.py classify (-t <file> | --) -m <mdl> -V <csv> [-e <encoding>]

Options:
    -c <csv>        Training .csv file.
    -C <str>        Classifier type [default: rf].
    -m <mdl>        Output .mdl file.
    -V <csv>        Output vocabulary .csv file.
    -t <file>       File containing incoming text to be classified. 
    -e <encoding>   File encoding [default: utf-8].
    --              Read incoming text from stdin.

"""

import os
import sys
import re
import codecs
import numpy
import pandas

from functools import reduce

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from docopt import docopt


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

COUNTER = 0


def read_blob(buf, encoding):
    if os.path.exists(buf):
        with open(buf, "rb") as f:
            str_val = codecs.getreader(encoding)(f).read()
    elif hasattr(buf, "read"):
        str_val = buf.read()
    elif isinstance(buf, basestring):
        str_val = buf
    else:
        raise NotImplementedError
    return str_val


def read_vocab(vocab_fn):
    df = pandas.read_csv(vocab_fn, index_col=None, header=None)
    return dict(df.to_records(index=None))



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


def remove_non_words(tokens):
    return ( t for t in tokens if valid_word(t) )


def remove_stop_words(tokens, stop_words=STOP_WORDS):
    return ( t for t in tokens if t not in stop_words )


def stem_words(tokens):
    return ( PORTER_STEMMER.stem(t) for t in tokens )


def normalize_tokens(tokens):
    return [ t.lower() for t in remove_non_words(
             remove_stop_words(stem_words(tokens))) ]


def preprocess_blob(blob):
    """Preprocess raw text (e.g. from a text file, html page, twitter stream
    etc.) by normalising and tokenizing into a list of more usable document
    terms.

    Arguments:
        blob --

    Returns:
        a single list of normalised terms associated with a single document
        blob.

    TODO:
        - filter markup tokens

    """
    tokens = reduce(lambda x, y : x + y,
                (word_tokenize(s) for s in sent_tokenize(blob.lower())))
    terms = normalize_tokens(tokens)

    return terms


def make_tfidf_matrix(docs, vocab=None):
    """
    """
    count_vectoriser = CountVectorizer(tokenizer=preprocess_blob, vocabulary=vocab)
    tfidf_transformer = TfidfTransformer(use_idf=True)

    X_counts = count_vectoriser.fit(docs)
    X_counts = count_vectoriser.transform(docs)

    X_tfidf = tfidf_transformer.fit(X_counts)
    X_tfidf = tfidf_transformer.transform(X_counts)

    return X_tfidf, count_vectoriser.vocabulary_


def train_model(clf_type, csv_fn, mdl_fn, vocab_fn):
    """
    """
    df = pandas.read_csv(csv_fn, index_col=None, header=None)
    X, Y = df[1], df[0]

    # Use the standard tf-idf matrix representation of the dataset.
    X_tfidf, X_vocab = make_tfidf_matrix(X)

    if clf_type == "rf":
        clf = RandomForestClassifier()
    else:
        raise NotImplementedError

    # Train the model
    clf.fit(X_tfidf.toarray(), Y)

    # Save the vocabulary to disk
    vocab = pandas.DataFrame(X_vocab.items())
    vocab.to_csv(vocab_fn, index=None, header=None)

    # Save (pickle) the model to disk
    joblib.dump(clf, mdl_fn)

    return clf, vocab


def classify_text(buf, mdl_fn, vocab_fn, encoding):
    """
    """

    blob = read_blob(buf, encoding=encoding)
    vocab = read_vocab(vocab_fn)
    docs = [blob]
    X_tfidf, X_vocab = make_tfidf_matrix(docs, vocab=vocab)
    clf = joblib.load(mdl_fn)

    # TODO:
    # - compare the model vocab with the incoming vocab.

    return clf.predict(X_tfidf.toarray())


if __name__ == "__main__":
    opts = docopt(__doc__)

    if opts["train"]:
        train_model(opts["-C"], opts["-c"], opts["-m"], opts["-V"])
    elif opts["classify"]:
        classify_text(opts["-t"], opts["-m"], opts["-V"], opts["-e"])
    else:
        raise NotImplementedError
