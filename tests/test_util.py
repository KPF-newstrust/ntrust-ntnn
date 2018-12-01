import numpy as np

from keras.preprocessing.text import Tokenizer
from ntnn import util
from ntnn.constant import categories


def test_should_vectorize_str():
    x = np.array(['가나a', '다다'])
    y = util.vectorize_str(x)
    assert y.shape == (2, 100)


def test_should_vectorize_category():
    cates = np.array(['사회', '정치'])
    y = util.vectorize_category(cates)
    expected = [categories[c] for c in cates]
    assert y.tolist() == expected


def test_should_convert_to_morphs():
    x = np.array(['스타벅스에서 커피를 마셨다.', 'asdf'])
    morphs, tags = util.to_morphs(x)

    assert morphs.shape == tags.shape
    assert morphs.shape[-1] == len(x)


def test_should_convert_to_sentences():
    x = np.array(['스타벅스에서 커피를 마셨다. 목이 아프다.'])
    y = util.to_sentences(x)

    assert y.shape == (2,)


def test_tokenizer():
    from keras.preprocessing.text import Tokenizer

    x = np.array(['스타 벅스 커피 마시다'])

    tokenizer = Tokenizer(num_words=10, filters='')
    tokenizer.fit_on_texts(x)
    ids = tokenizer.texts_to_sequences(x)
    ids = np.array(ids, dtype=np.int32)


def test_should_convert_to_tfidf():
    x = np.array(['청와대 대통령 정치'])

    tokenizer = Tokenizer(num_words=100, filters='')
    tokenizer.fit_on_texts(x)

    seqlen = 10
    y = util.to_tfidf(tokenizer, x, seqlen=seqlen)
    assert y.shape == (1, seqlen)
