import asyncio
from konlpy.tag import Mecab
import numpy as np
import os
import sys
from tensorflow.contrib.predictor import from_saved_model
from tensorflow.contrib.learn import preprocessing

from ntnn.constant import categories


def preproc(mecab, vocab, str):
    poss = mecab.pos(str)
    allowed = ['NNP', 'NNG', 'VV', 'VA']
    selected = [p[0] for p in poss if p[1] in allowed]

    ids = vocab.transform([' '.join(selected)])
    return {'x': np.array(np.array(list(ids)), dtype=np.int64)}


def load_vocab(vocab_dir):
    datafile = os.path.join(vocab_dir, 'vocab.db')

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(__dir__, '../../utils'))

    vocab = preprocessing.VocabularyProcessor.restore(datafile)
    vocab.vocabulary_.freeze(False)

    sys.path.pop()

    return vocab


class Classifier:
    def __init__(self, model_dir):
        self.mecab = Mecab()
        self.vocab = load_vocab(model_dir)
        self.predictor = from_saved_model(export_dir=model_dir)

    async def predict(self, val):
        loop = asyncio.get_event_loop()

        features = await loop.run_in_executor(
            None, preproc, self.mecab, self.vocab, val)

        cls = await loop.run_in_executor(
            None, self.predictor, features)

        return next(
            name for name, id in categories.items()
            if id == cls['class'][0])
