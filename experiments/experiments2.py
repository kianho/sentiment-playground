#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Tue Feb 24 12:10:56 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    TODO

Usage:
    kaggle-eda.py -i <csv>

Options:
    -i <csv>


"""

import os
import sys
import pprint
import re

import numpy
import pandas
import mpl_preamble
import pylab

from sentiment import make_term_matrix, tokenize, STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.lda import LDA
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import scale

from docopt import docopt

numpy.random.seed(12345611)

RE_NT = re.compile(r"([a-zA-Z]+)\s+(n't)")

TEST = \
pandas.Series(["But it does n't leave you with much .",
 "While The Importance of Being Earnest offers opportunities for occasional smiles and chuckles , it does n't give us a reason to be in the theater beyond Wilde 's wit and the actors ' performances .",
 "Do n't judge this one too soon - it 's a dark , gritty story but it takes off in totally unexpected directions and keeps on going .",
 "I 'm sure the filmmaker would disagree , but , honestly , I do n't see the point .",
 "This is a heartfelt story ... it just is n't a very involving one .",
 "Report card : Does n't live up to the exalted tagline - there 's definite room for improvement .",
 "Takes one character we do n't like and another we do n't believe , and puts them into a battle of wills that is impossible to care about and is n't very funny .",
 "Although fairly involving as far as it goes , the film does n't end up having much that is fresh to say about growing up Catholic or , really , anything .",
 "The movie is silly beyond comprehension , and even if it were n't silly , it would still be beyond comprehension .",
 "Like the world of his film , Hartley created a monster but did n't know how to handle it ."])

if __name__ == '__main__':
    opts = docopt(__doc__)


    #print TEST.apply(lambda s : RE_NT.sub(r"\1\2", s))

    df = pandas.read_csv(opts["-i"])

    count_vectoriser = CountVectorizer(stop_words="english", ngram_range=(1,2))
    tfidf_transformer = TfidfTransformer(use_idf=True)

    X = count_vectoriser.fit_transform(df.Phrase)
    X = tfidf_transformer.fit_transform(X)
    y = df.Sentiment

#   term2index = count_vectoriser.vocabulary_
#   index2term = dict((v, k) for (k, v) in term2index.items())

#   chi2_vals, p_vals = chi2(X, y)

#   ranks = \
#           pandas.DataFrame([(index2term[i], i, chi2_vals[i], p_vals[i])
#               for i in numpy.argsort(chi2_vals)[::-1]],
#               columns=["term", "index", "chi2", "pval"])

#   vocab_best = dict((r["term"], r["index"]) for (_, r) in ranks.iloc[:3000].iterrows())

#   count_vectoriser = CountVectorizer(vocabulary=vocab_best.keys(), ngram_range=(2,2))
#   tfidf_transformer = TfidfTransformer(use_idf=True)

#   X = count_vectoriser.fit_transform(df.Phrase)
#   X = tfidf_transformer.fit_transform(X)
#   y = df.Sentiment

#   print len(vocab_best), X.shape

    cv = StratifiedKFold(y, n_folds=2)
    #X = scale(X.toarray())
    #X = X.toarray()

    for train_ix, test_ix in cv:
        clf = MultinomialNB()
        #clf = OneVsOneClassifier(SGDClassifier(loss="log", penalty="l2"))
        #clf = OneVsOneClassifier(GaussianNB())
        #clf = LDA()
        clf.fit(X[train_ix], y[train_ix])
        print clf.predict(X[test_ix])
        print clf.score(X[test_ix], y[test_ix])

    #   cv_scores = cross_val_score(MultinomialNB(),
    #           X=X, y=y, cv=cv, scoring="accuracy", n_jobs=1, verbose=3)

    #   print cv_scores

    sys.exit()

    ################################

    sys.exit()

    df2 = df.groupby("SentenceId").agg(lambda d : d.iloc[0])
    df2.Phrase = df2.Phrase.apply(lambda p : RE_NT.sub(r"\1\2", p.lower()))

    count_vect = CountVectorizer(tokenizer=tokenize,
            stop_words="english", binary=False)
    X = count_vect.fit_transform(df2.Phrase)
    y = df2.Sentiment

    term2index = count_vect.vocabulary_
    index2term = dict((v, k) for (k, v) in term2index.items())

    chi2_vals, p_vals = chi2(X, y)

    ranks = \
            pandas.DataFrame([(index2term[i], i, chi2_vals[i], p_vals[i])
                for i in numpy.argsort(chi2_vals)[::-1]],
                columns=["term", "index", "chi2", "pval"])

    vocab_best = dict((r["term"], r["index"]) for (_, r) in ranks[ranks["pval"] < 0.05].iterrows())

    count_vect = CountVectorizer(tokenizer=tokenize,
            vocabulary=vocab_best.keys(), stop_words="english", binary=False)

    X = count_vect.fit_transform(df2.Phrase)

    clf = RandomForestClassifier(n_estimators=500, n_jobs=4, verbose=2, oob_score=True)
    clf.fit(X.toarray(), y)
    print clf.oob_score_

    sys.exit()

    sys.exit()

    X, vocab = make_term_matrix(df2.Phrase, toarray=False,
            tfidf=False,
            stop_words=STOP_WORDS,
        preprocessor=lambda p : RE_NT.sub(r"\1\2", p.lower()))


    y = df2.Sentiment
    #X = scale(X)

    #print X.shape
    #print y.shape

    print X
    print y

#   clf = RandomForestClassifier(n_estimators=500, n_jobs=4, verbose=2, oob_score=True)
#   #clf = MultinomialNB()
#   clf.fit(X.toarray(), y)

#   print clf.oob_score_

    cv = StratifiedKFold(y, n_folds=3)
    cv_scores = cross_val_score(LDA(),
            X=X.toarray(), y=y, cv=cv, scoring="accuracy", n_jobs=1, verbose=3)
    print cv_scores

#   cv = StratifiedKFold(y, n_folds=5)
#   cv_scores = cross_val_score(RandomForestClassifier(n_estimators=500),
#           X=X, y=y, cv=cv, scoring="accuracy", n_jobs=2, verbose=1)

#   print cv_scores

    sys.exit()
    #df["tokens"] = df["Phrase"].str.lower().apply(lambda p : " ".join(tokenize(p)))


    df_stop = df[(df["tokens"].isin(STOP_WORDS)) & df["tokens"].str.len() == 1]

    print df_stop.Sentiment.value_counts(True, True)
    #pylab.hist(df_stop["Sentiment"].tolist())
    #pylab.savefig("blah.pdf")

    sys.exit()

    #
    # no. of terms vs. sentiment.
    #
    df_neutral = df[df.Sentiment >= 0]

    df_neutral["n_tokens"] = \
            df_neutral.Phrase.apply(lambda p : len(sentiment.tokenize(p, stop_words=sentiment.STOP_WORDS)))

    pylab.hist(df_neutral["n_tokens"].tolist())
    pylab.savefig("blah.pdf")
