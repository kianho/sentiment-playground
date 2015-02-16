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
    """
    """
    return not (
        RE_NON_WORD_CHARS.match(w) or
        RE_MULTI_CHAR_PUNCT.match(w) or
        RE_NUMBER.match(w) or
        RE_SINGLE_NON_ALPHA.match(w)
    )


def my_tokenizer(s):
    """Trivial whitespace tokenizer since the polarity 2.0 data is already
    pre-tokenized.

    """
    return [ t for t in RE_SPACE.split(s) ]
