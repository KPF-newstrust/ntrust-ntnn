import re

import numpy as np

from functools import partial
from gensim.models.callbacks import CallbackAny2Vec
from konlpy.tag import Mecab
from ntnn.constant import categories
from ntnn.constant import token as T


tagger = Mecab()

split = partial(re.split, r'\s+')


def vectorize_category(x, defaults=0, dtype=np.int8):
    def fn(s):
        return categories[s] if s in categories else defaults
    return np.vectorize(fn, otypes=[dtype])(x)


def vectorize_str(x, maxlen=100, dtype=np.int16):
    """
    maxlen: max character length. hangul = 3 chars
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

    y = np.zeros((len(x), maxlen), dtype=dtype)
    for i, s in enumerate(x):
        ids = []
        for c in s:
            ids += ch2ids(c)

        ln = min(maxlen, len(ids))
        y[i, :ln] = ids[:ln]
    return y


def to_filtered(x):
    def fn(s):
        assert type(s) is str

        s = re.sub(r'\[[^\]]+\]', '', s)  # [*] 제외
        s = re.sub(r'\([^\)]+\)', '', s)  # (*) 제거
        s = re.sub(r'\<[^\>]+\>', '', s)  # <*> 제거
        s = re.sub(r'[^\u0020-\u007E\uAC00-\uD7AF]', ' ', s)
        s = re.sub(r'\\n', '\n', s)
        s = re.sub(r'\s{2,}', ' ', s)
        s = s.lower()
        return s
    return np.vectorize(fn)(x)


def to_morphs(x, includes=None):
    y = np.full((len(x),), '', dtype=object)
    yt = np.full((len(x),), '', dtype=object)

    for i, s in enumerate(x):
        morphs, tags = [], []
        for morph, tag in tagger.pos(s):
            if includes and tag not in includes:
                continue

            morphs.append(morph)
            tags.append(tag)

        y[i] = ' '.join(morphs)
        yt[i] = ' '.join(tags)
    return y, yt


def to_sentences(docs, includes=None):
    brackets = {
        '(': ')',
        '[': ']',
        '<': '>',
        '"': '"”“',
        '“': '"”“',
        '”': '"”“',
        "'": "'’‘",
        "’": "'’‘",
        "‘": "'’‘"}
    EOS = '.!?\n\\n'

    y = []
    for doc in docs:
        sentence = []
        needed_brackets = ''
        for ch in doc:
            if len(needed_brackets):
                if ch in needed_brackets:
                    needed_brackets = ''
                sentence.append(ch)
            else:
                if ch in brackets:
                    needed_brackets = brackets[ch]
                sentence.append(ch)

                if ch in EOS:
                    if len(sentence):
                        y.append(''.join(sentence).strip())
                    sentence = []

        if len(sentence):
            y.append(''.join(sentence).strip())

    return np.array(y, dtype=object)


def to_tfidf(tokenizer, texts, seqlen=100, dtype=np.float32):
    seqs = tokenizer.texts_to_sequences(texts)
    mat = tokenizer.texts_to_matrix(texts, mode='tfidf')

    y = np.zeros((len(seqs), seqlen), dtype=dtype)
    for i, seq in enumerate(seqs):
        for j, idx in enumerate(seq[:seqlen]):
            y[i, j] = mat[i, idx]
    return y


class EpochLogger(CallbackAny2Vec):
    def __init__(self, total):
        self.total = total
        self.epoch = 1

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        print("Trained %d/%d" % (self.epoch, self.total))
        self.epoch += 1
