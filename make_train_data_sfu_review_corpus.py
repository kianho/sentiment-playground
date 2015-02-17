#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Mon Feb 16 19:44:03 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    TODO

Usage:
    make_train_data_sfu_review_corpus.py -l LABEL

Options:
    -l LABEL

"""

import sys
import codecs
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from docopt import docopt
from ftfy import fix_text
from utils import normalize_tokens


if __name__ == '__main__':
    opts = docopt(__doc__)

    records = []

    for fn in ( ln.strip() for ln in sys.stdin ):
        with open(fn) as f:
            # Some files had a weird encoding, which made it incompatible with
            # the nltk tokenizers. Therefore, I used the ftfy package to
            # infer/fix the encodings of the file before proceeding with the
            # standard tokenization process.
            #
            # NOTE: need to read the raw data files using the "cp1252" decoding
            # for the sfu review corpus.
            blob = fix_text(codecs.getreader("cp1252")(f).read())

            all_tokens = []
            for sentence in sent_tokenize(blob):
                tokens = word_tokenize(sentence)
                tokens = normalize_tokens(tokens)
                all_tokens.extend(tokens)

            records.append((int(opts["-l"]), " ".join(all_tokens)))

    df = pd.DataFrame.from_records(records, columns=["label", "text"])
    df.to_csv(sys.stdout, index=None, encoding="utf-8")
