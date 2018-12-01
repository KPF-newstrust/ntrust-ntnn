import re

import numpy as np

from gensim.models.callbacks import CallbackAny2Vec
from konlpy.tag import Mecab
from ntnn.constant import categories
from ntnn.constant import token as T


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

    def vec(s):
        ids = []
        for c in s:
            ids += ch2ids(c)
        return ids

    y = np.zeros((len(x), maxlen), dtype=dtype)
    for i, s in enumerate(x):
        x_ = vec(s)
        ln = min(maxlen, len(x_))
        y[i, :ln] = x_[:ln]
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
    def convert(x):
        tagger = Mecab()
        morphs, tags = [], []
        for morph, tag in tagger.pos(x):
            if includes and tag not in includes:
                continue

            morphs.append(morph)
            tags.append(tag)
        return ' '.join(morphs), ' '.join(tags)

    morphs = np.full((len(x),), '', dtype=object)
    tags = np.full((len(x),), '', dtype=object)
    for i, s in enumerate(x):
        morph, tag = convert(s)
        morphs[i] = morph
        tags[i] = tag

    return morphs, tags


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
    EOS = ['.', '!', '?', '\n', '\\n']

    def convert(doc):
        sentences = []
        sentence = ''
        needed_brackets = []
        for ch in doc:
            if len(needed_brackets):
                if ch in needed_brackets:
                    needed_brackets = []
                sentence += ch
            else:
                if ch in brackets:
                    needed_brackets = [*brackets[ch]]
                sentence += ch

                if ch in EOS:
                    if len(sentence.strip()):
                        sentences.append(sentence)
                    sentence = ''

        if len(sentence.strip()):
            sentences.append(sentence)
        return sentences

    y = []
    for s in docs:
        y = y + convert(s)
    return np.array(y)


def to_tfidf(tokenizer, texts, seqlen=100, dtype=np.float32):
    seqs = tokenizer.texts_to_sequences(texts)
    mat = tokenizer.texts_to_matrix(texts, mode='tfidf')

    y = np.zeros((len(seqs), seqlen), dtype=dtype)
    for i, seq in enumerate(seqs):
        for j, idx in enumerate(seq[:seqlen]):
            y[i, j] = mat[i, idx]
    return y


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        print(".")
        self.epoch += 1
