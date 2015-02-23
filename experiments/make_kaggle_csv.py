#!/usr/bin/env python
# encoding: utf-8
"""

Date: Mon Feb 23 16:51:22 AEDT 2015
Author: Kian Ho <hui.kian.ho@gmail.com>

Description:
    ...

Usage:
    make_train_data_kaggle.py -i <tsv> -o <csv>

Options:
    -i <tsv>
    -o <csv>

"""

import os
import sys
import pandas

from docopt import docopt


if __name__ == '__main__':
    opts = docopt(__doc__)
    pandas.read_table(opts["-i"], sep="\t").to_csv(opts["-o"], index=None)
