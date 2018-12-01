import numpy as np

from ntnn.clustering import select_hangul


def test_should_select_hangul():
    x = np.array(['starbucks 에 가다 .'])
    tags = np.array(['a b c d'])

    y, ty = select_hangul(x, tags)
    assert y[0] == '에 가다'
    assert ty[0] == 'b c'
