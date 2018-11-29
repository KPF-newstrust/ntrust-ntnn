import numpy as np

from ntnn import util
from ntnn.constant import categories


def test_should_vectorize_str():
    x = np.array(['가나a'], dtype=np.str)
    y = util.vectorize_str(x)
    assert y.shape == (1, 100)
    assert np.trim_zeros(y[0]).size == 7


def test_should_vectorize_category():
    cates = np.array(['사회', '정치'], dtype=np.str)
    y = util.vectorize_category(cates)
    expected = np.array([categories[c] for c in cates], dtype=np.int16)
    assert np.array_equal(y, expected)


def test_should_convert_to_morphs():
    x = ['스타벅스에서 커피를 마셨다.']
    morphs, tags = util.to_morphs(x)

    assert morphs.shape == tags.shape
    assert morphs.shape[-1] == 1


def test_one_hot():
    x = np.array(['스타 벅스 커피 마시다'])

    y = util.to_one_hot(x, nclasses=100, maxlen=10)
    assert y.shape == (1, 10, 100)


def test_tokenizer():
    from keras.preprocessing.text import Tokenizer

    x = np.array(['스타 벅스 커피 마시다'])

    tokenizer = Tokenizer(num_words=10, filters='')
    tokenizer.fit_on_texts(x)
    ids = tokenizer.texts_to_sequences(x)
    ids = np.array(ids, dtype=np.int32)
