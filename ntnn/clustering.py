import re

import numpy as np


def is_hangul(word):
    return re.match(r'^[\uAC00-\uD7AF]+$', word) is not None


def _select_hangul(words, tags):
    S, T = [], []
    for i, w in enumerate(words):
        if is_hangul(w):
            S.append(w)
            T.append(tags[i])
    return ' '.join(S), ' '.join(T)


def select_hangul(sentences, tags):
    S = np.full((len(sentences),), '', dtype=object)
    T = np.full((len(tags),), '', dtype=object)
    for i, (sentence, tag) in enumerate(zip(sentences, tags)):
        sentence, tag = sentence.split(' '), tag.split(' ')
        S[i], T[i] = _select_hangul(sentence, tag)
    return S, T
