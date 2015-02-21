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
    sentiment.py train -c <csv> [-C <str>] -m <mdl> [-e <encoding>] [<param> ...]
    sentiment.py classify (-t <file> | --) -m <mdl> [-e <encoding>]

Options:
    -c <csv>        Training .csv file.
    -C <str>        Valid sklearn classifier [default: RandomForestClassifier].
    -m <mdl>        Output .mdl file.
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

STOP_WORDS = set(stopwords.words("english"))
PORTER_STEMMER = PorterStemmer()

RE_ALPHA = re.compile(r"[a-zA-Z]+")
RE_PUNCT = re.compile(r"""[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]""")
RE_PUNCT_WORD = re.compile(r"^%s+$" % RE_PUNCT.pattern)
RE_NON_WORD = re.compile(r"((?:[?!)\";}\]\*:@\'\({\[]))")
RE_RATING = re.compile(r"[0-9]+/[0-9]+")
RE_NUMBER_ONLY = re.compile(r"^(\d+\.?\d*)$")

# Tokens that match this regular expression will be ignored.
NON_TOKEN_RE = re.compile(r"""
    # Non-word characters, taken from nltk.tokenize.punkt.
    ((?:[?!)\";}\]\*:@\'\({\[]))
    |
    # Multi-character punctuation, e.g. ellipses, en-, and em-dashes.
    (?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)
    |
    # Integer or floating-point number.
    (\d+\.?\d*")
    |
    # Single non-alphabetic character.
    (^[^a-zA-Z]$)
""")


def read_blob(buf, encoding="utf-8"):
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


# TODO: remove this function.
def read_vocab(vocab_fn):
    df = pandas.read_csv(vocab_fn, index_col=None, header=None)
    return dict(df.to_records(index=None))


def load_model(mdl_fn):
    mdl = joblib.load(mdl_fn)
    clf, vocabulary = mdl
    return clf, vocabulary


def parse_clf_kwargs(params):
    """Parse the classifier constructor keyword arguments specified from a list
    of "<key>=<value>" strings. These values are typically supplied at the
    command-line.

    """
    def parse_val(s):
        val = s

        if s in set(("True", "False")):
            val = (s == "True")
        else:
            for _type in (int, float):
                try:
                    val = _type(s)
                    break
                except ValueError:
                    continue

        return val

    clf_kwargs = {}
    for k, s in (p.split("=") for p in params):
        clf_kwargs[k] = parse_val(s)

    return clf_kwargs


def make_clf(clf_name, params):
    """
    """
    import sklearn

    clf_kwargs = parse_clf_kwargs(params)

    return eval(clf_name)(**clf_kwargs)


def valid_token(t, stop_words=None):
    if stop_words is None:
        stop_words = set()
    return ((not NON_TOKEN_RE.match(t))
            and RE_ALPHA.match(t) and (t.lower() not in stop_words))


def remove_non_words(tokens):
    return ( t for t in tokens if valid_token(t) )


def remove_stop_words(tokens, stop_words=STOP_WORDS):
    return ( t for t in tokens if t not in stop_words )


def stem_words(tokens):
    return ( PORTER_STEMMER.stem(t) for t in tokens )


def normalize_tokens(tokens):
    return [ t.lower() for t in remove_non_words(
             remove_stop_words(stem_words(tokens))) ]


def tokenize(blob, stem_func=None, stop_words=None):
    """Simple tokenizer that removes tokens based on a number of regular
    expression patterns. Users may optionally use a stemming function and/or a
    set of stop words to omit from the final token list.

    """
    tokens = reduce(lambda a, b : a + b, 
                (word_tokenize(s) for s in sent_tokenize(blob)))

    tokens = ( t for t in tokens
                if (
                    # remove tokens containing only punctuations.
                    (not RE_PUNCT_WORD.match(t)) and

                    # remove tokens containing illegal characters.
                    (not RE_NON_WORD.search(t)) and 

                    # remove tokens containing (potential) ratings (e.g. 8/10).
                    (not RE_RATING.search(t)) and

                    # remove tokens containing only an integer/float.
                    (not RE_NUMBER_ONLY.match(t))
                )
            )

    if stem_func:
        tokens = ( stem_func(t) for t in tokens )

    if stop_words:
        tokens = ( t for t in tokens if t not in stop_words )

    return list(tokens)


def make_tfidf_matrix(docs, toarray=True, **count_vect_kwargs):
    """
    """

    # Set the default count_vect_kwargs values.
    count_vect_kwargs.setdefault("tokenizer", tokenize)

    count_vectoriser = CountVectorizer(**count_vect_kwargs)
    tfidf_transformer = TfidfTransformer(use_idf=True)

    X_counts = count_vectoriser.fit(docs)
    X_counts = count_vectoriser.transform(docs)

    X_tfidf = tfidf_transformer.fit(X_counts)
    X_tfidf = tfidf_transformer.transform(X_counts)

    if toarray:
        X_tfidf = X_tfidf.toarray()

    return X_tfidf, count_vectoriser.vocabulary_


def train_model(clf, csv_fn, mdl_fn):
    """
    """
    df = pandas.read_csv(csv_fn, index_col=None, header=None)
    X, Y = df[1], df[0]

    # Use the standard tf-idf matrix representation of the dataset.
    X_tfidf, X_vocab = make_tfidf_matrix(X, tokenizer=tokenize)

    # Train the model
    clf.fit(X_tfidf, Y)

    # Save the classifier and vocabulary to disk i.e. as a single model (mdl)
    # file.
    mdl = (clf, X_vocab)
    joblib.dump(mdl, mdl_fn)

    return mdl


def classify_text(buf, mdl_fn, encoding):
    """
    """
    blob = read_blob(buf, encoding=encoding)
    clf, vocab = joblib.load(mdl_fn)
    X, X_vocab = make_tfidf_matrix([blob], vocabulary=vocab)
    X = X.toarray() # Some classifiers can't handle sparse matrices.

    # NOTE: predict_proba() and predict() each return an array.
    y_pred, y_score = clf.predict_proba(X)[0][1], clf.predict(X)[0]

    return y_pred, y_score


if __name__ == "__main__":
    opts = docopt(__doc__)

    if opts["train"]:
        clf_kwargs = parse_clf_kwargs(opts["<param>"])
        clf = eval(opts["-C"])(**clf_kwargs)
        train_model(clf, opts["-c"], opts["-m"])
    elif opts["classify"]:
        blob = read_blob(opts["-t"])
        y_pred, y_score = classify_text(blob, opts["-m"], opts["-e"])
        print y_pred, y_score
    else:
        raise NotImplementedError
