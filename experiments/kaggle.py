#!/usr/bin/env python
# encoding: utf-8
"""

Date:
    Mon Feb 23 18:50:10 AEDT 2015

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    TODO

Usage:
    kaggle.py -t <csv>

Options:
    -t <csv>    Kaggle training/development data.

"""

import os
import sys

from docopt import docopt


if __name__ == '__main__':
    opts = docopt(__doc__)
