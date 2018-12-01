import re
import numpy as np

from ntnn.util import split


def is_hangul(word):
    return re.match(r'^[\uAC00-\uD7AF]+$', word) is not None


def select_hangul(sentences, tags):
    S = np.full((len(sentences),), '', dtype=object)
    T = np.full((len(tags),), '', dtype=object)
    for i, (sentence, tag) in enumerate(zip(sentences, tags)):
        ss, ts = [], []

        for w, t in zip(split(sentence), split(tag)):
            if is_hangul(w):
                ss.append(w)
                ts.append(t)
        S[i] = ' '.join(ss)
        T[i] = ' '.join(ts)
    return S, T
