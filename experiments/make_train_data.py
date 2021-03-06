#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Sat Feb 14 17:13:48 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    Read a list of paths to text files from stdin and concatenate their contents
    into a minimal-quoted csv format.

Usage:
    make_train_data.py --label LABEL [--csv CSV] [--encoding ENCODING]

Options:
    -l, --label LABEL           Label to assign each review instance.
    -c, --csv CSV               Write output to .csv file.
    -e, --encoding ENCODING     Input file encoding [default: utf-8].

Example workflow:
    ...

"""

import os
import sys
import re
import codecs
import pandas

from docopt import docopt

if __name__ == '__main__':
    opts = docopt(__doc__)

    label = int(opts["--label"])
    records = []
    for fn in ( ln.strip() for ln in sys.stdin ):
        with open(fn) as f:
            blob = codecs.getreader(opts["--encoding"])(f).read()
        records.append((label, blob))

    df = pandas.DataFrame.from_records(records, columns=None)
    df.to_csv(opts["--csv"] if opts["--csv"] else sys.stdout,
            index=None, header=False, encoding="utf-8")
