import re

import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.layers.core import Dense, Activation, Dropout, Flatten

from ntnn.constant import token as T, categories


def to_filtered(x):
    def fn(s):
        s = re.sub(r'\[[^\]]+\]', '', s)  # [*] 제외
        s = re.sub(r'\([^\)]+\)', '', s)  # (*) 제거
        s = re.sub(r'\<[^\>]+\>', '', s)  # <*> 제거
        s = re.sub(r'[^\u0020-\u007E\uAC00-\uD7AF]', '', s)
        s = re.sub(r'\\n', '\n', s)
        s = re.sub(r'\s{2,}', ' ', s)
        return s
    return np.vectorize(fn, otypes=[np.str])(x)


def build_model(input_dim=T.Total, out_dim=len(categories), seqlen=1000):
    model = Sequential()
    model.add(Embedding(input_dim, 64, input_length=seqlen))
    model.add(Conv1D(64, 5, strides=1, activation='relu'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(3, strides=2))
    model.add(Conv1D(64, 20, strides=4))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(3, strides=2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.25))
    model.add(Dense(out_dim))
    model.add(Activation('relu'))
    return model
