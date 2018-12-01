import numpy as np


def eval_w2v(word2vec, positives, negatives=[]):
    positives = [w for w in positives if w in word2vec.wv]
    negatives = [w for w in negatives if w in word2vec.wv]

    if not positives and not negatives:
        return ''

    return word2vec.wv.most_similar(
        positive=positives, negative=negatives)


def to_w2v_texts(w2v, texts, tags):
    y = np.full((len(texts),), '', dtype=object)
    yt = np.full((len(tags),), '', dtype=object)

    for i, (text, tag) in enumerate(zip(texts, tags)):
        valid_texts = []
        valid_tags = []
        for w, t in zip(text.split(' '), tag.split(' ')):
            if w in w2v.wv:
                valid_texts.append(w)
                valid_tags.append(t)
        y[i] = ' '.join(valid_texts)
        yt[i] = ' '.join(valid_tags)
    return y, yt


def text_to_tfidf(tokenizer, text):
    assert type(text) is str

    seqs = tokenizer.texts_to_sequences([text])
    mat = tokenizer.texts_to_matrix([text], mode='tfidf')
    assert len(seqs) == 1

    y = np.full((len(text.split(' ')),), 1.0, dtype=np.float32)
    for i, s in enumerate(seqs[0]):
        y[i] = mat[0, s]
    return y


def to_word_vectors(w2v, texts, size=100, tfidfer=None, dtype=np.float32):
    y = np.zeros((len(texts), size), dtype=dtype)

    for i, text in enumerate(texts):
        words = text.split(' ')
        v = np.zeros((len(words), size), dtype=dtype)

        tfidf = (
            text_to_tfidf(tfidfer, text) if tfidfer
            else np.full((len(words),), 1.0, dtype=np.float32))

        for j, w in enumerate(words):
            if w in ['']: continue
            v[j] = w2v.wv[w] * tfidf[j]
        y[i] = np.mean(v, axis=0)
    return y


class StrCorpus(object):
    def __init__(self, series, separator=' '):
        self.series = series
        self.separator = separator

    def __iter__(self):
        for i, doc in enumerate(self.series):
            assert type(doc) is str
            yield doc.split(self.separator)

    def __len__(self):
        return len(self.series)
