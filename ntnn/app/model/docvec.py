import os
from konlpy.tag import Mecab
from clustering_sample.cluster_batch import Word2Vec, strip, morps


class Docvec:
    def __init__(self, model_dir):
        self.mecab = Mecab()
        self.w2v = self._w2v(model_dir)

    def _w2v(self, model_dir):
        w2vfile = os.path.join(model_dir, 'wv.model')
        assert os.path.exists(w2vfile), 'Invalid model path for w2v'
        return Word2Vec(w2vfile)

    def _preproc(self, str)
        str = strip(str)
        return morps(str, self.mecab)

    def from_text(self, str):
        str = self._preproc(str)
        return self.w2v.transform_docs_by_tfidf([str])

