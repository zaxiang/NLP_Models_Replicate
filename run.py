#!/usr/bin/env python

import sys
import json

from runner import *


def main(targets):
	TFIDF_runner('nyt', 'coarse')
	W2V_Runner('nyt', 'coarse')

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)