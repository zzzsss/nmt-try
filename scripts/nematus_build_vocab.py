#!/usr/bin/env python2

# from nematus/data/build_dictionary.py, but using stdio instead of files

import numpy
import json
import sys
from collections import OrderedDict

def main():
    word_freqs = OrderedDict()
    for line in sys.stdin:
        words_in = line.strip().split(' ')
        for w in words_in:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1
    words = word_freqs.keys()
    freqs = word_freqs.values()
    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]
    worddict = OrderedDict()
    worddict['eos'] = 0
    worddict['UNK'] = 1
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii+2
    print(json.dumps(worddict, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
