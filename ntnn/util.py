import numpy as np

from keras.preprocessing.text import one_hot
from konlpy.tag import Mecab
from ntnn.constant import categories
from ntnn.constant import token as T


def vectorize_category(x, defaults=0, dtype=np.int8):
    def fn(s):
        return categories[s] if s in categories else defaults
    return np.vectorize(fn, otypes=[dtype])(x)


def vectorize_str(x, maxlen=100, dtype=np.int16):
    """
    maxlen: max character length. 1 hangul = 3 chars
    """
    def ch2ids(ch):
        n = ord(ch)
        if n in T.Hangul:
            han = n - T.Hangul[0]
            cho = int(han / T.Jong / T.Jung) + T.ChoId
            jung = int(han / T.Jong) % T.Jung + T.JungId
            jong = int(han % T.Jong) + T.JongId
            return [cho, jung, jong]

        if n in T.Ascii:
            return [n - T.Ascii[0] + T.AsciiId]

        if n in T.Symbol:
            return [n - T.Symbol[0] + T.SymbolId]

        if n in T.Symbol2:
            return [n - T.Symbol2[0] + T.Symbol2Id]
        return [T.UnkId]

    def str_fn(s):
        ids = []
        for c in s:
            ids += ch2ids(c)
        padded = ids[:maxlen] + [0] * (maxlen - len(ids))
        return np.array(padded, dtype=dtype)

    str2ids = np.frompyfunc(str_fn, 1, 1)
    return np.vstack(str2ids(x))


def to_morphs(x, includes=None):
    def fn(x):
        tagger = Mecab()
        morphs, tags = [], []
        for morph, tag in tagger.pos(x):
            if includes and tag not in includes:
                continue

            morphs.append(morph)
            tags.append(tag)
        return ' '.join(morphs), ' '.join(tags)

    vectorize = np.frompyfunc(fn, 1, 2, otypes=[np.str])
    return vectorize(x)


def to_one_hot(x, nclasses=100, maxlen=100, dtype=np.int16):
    def fn(s):
        y = one_hot(s, nclasses)
        y = y[:maxlen] + [0] * (maxlen - len(y))
        return np.array(y, dtype=np.int16)

    vectorize = np.frompyfunc(fn, 1, 1, otypes=[np.str])
    y = vectorize(x)
    y = np.vstack(y)
    y = np.eye(nclasses)[y]
    return y
