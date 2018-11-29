import numpy as np

from ntnn.classification_cnn import to_filtered, build_model


def test_should_filter_str():
    x = np.array(['가나<x>(d)a'], dtype=np.str)
    y = to_filtered(x)
    assert y[0] == '가나a'


def test_should_build_model():
    x = np.random.randint(1000, size=(10, 100))

    model = build_model(1000, 32, seqlen=100)
    model.compile('adam', loss='categorical_crossentropy')

    y = model.predict(x)
    assert y.shape == (10, 32)
