import numpy as np

from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from ntnn.word2vec import StrCorpus
from ntnn.word2vec import to_w2v_texts, to_word_vectors, text_to_tfidf


def test_should_convert_to_w2v_texts():
    data = np.array(['청와대 대통령 정치'], dtype=object)
    test = np.array(['청와대 대통령 강남'], dtype=object)

    train = StrCorpus(data)
    w2v = Word2Vec(train, min_count=1)

    y, tag = to_w2v_texts(w2v, test, test)
    assert y.shape == tag.shape
    assert y.shape == (1,)
    assert len(y[0].split(' ')) == 2


def test_should_convert_to_word_vectors():
    wvsize = 10
    data = np.array(['청와대 대통령 정치'], dtype=object)
    test = np.array(['청와대 대통령'], dtype=object)

    train = StrCorpus(data)
    w2v = Word2Vec(train, size=wvsize, min_count=1)

    y = to_word_vectors(w2v, test, size=wvsize)
    assert y.shape == (1, wvsize)


def test_should_convert_to_tfidf():
    x = np.array(['청와대 대통령 정치'], dtype=object)

    tokenizer = Tokenizer(num_words=100, filters='')
    tokenizer.fit_on_texts(x)

    y = text_to_tfidf(tokenizer, x[0])
    assert len(y) == len(x[0].split(' '))


def test_should_convert_to_word_vectors_with_tfidf():
    wvsize = 10
    data = np.array(['청와대 대통령 정치'], dtype=object)
    test = np.array(['청와대 대통령'], dtype=object)

    tokenizer = Tokenizer(num_words=100, filters='')
    tokenizer.fit_on_texts(data)

    train = StrCorpus(data)
    w2v = Word2Vec(train, size=wvsize, min_count=1)

    y = to_word_vectors(w2v, test, size=wvsize, tfidfer=tokenizer)
    assert y.shape == (1, wvsize)
