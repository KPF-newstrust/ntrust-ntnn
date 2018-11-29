import re

import numpy as np


def to_filtered(x):
    def fn(s):
        s = re.sub(r'\[[^\]]+\]', '', s)  # [*] 제외
        s = re.sub(r'\([^\)]+\)', '', s)  # (*) 제거
        s = re.sub(r'\<[^\>]+\>', '', s)  # <*> 제거
        s = re.sub(r'[^\uAC00-\uD7AF]', '', s)
        s = re.sub(r'\s{2,}', ' ', s)
        return s
    return np.vectorize(fn, otypes=[np.str])(x)
