import argparse
import csv
import os
import pandas as pd

from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import DBSCAN
from keras.preprocessing.text import Tokenizer
from ntnn.util import to_morphs, to_filtered
from ntnn.clustering import select_hangul
from ntnn.word2vec import to_w2v_texts, to_word_vectors


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--workdir', default='./.works/clustering')
parser.add_argument('--version', default=1)
parser.add_argument('--w2v_version', default='sg1')
parser.add_argument('--vocabsize', default=100000)
parser.add_argument(
    '--eps', default=0.5, help='neighborhood로 인정되는 최대 거리')
parser.add_argument('--coresize', default=3)
parser.add_argument('--nrows', default=None, type=int)

flag = parser.parse_args()
chunksize = 10000


def read_data():
    return pd.read_csv(
        os.path.join(flag.workdir, 'train.csv'),
        nrows=flag.nrows,
        header=0,
        delimiter='|',
        skipinitialspace=True,
        quoting=csv.QUOTE_MINIMAL)


def preproc(data, w2v):
    x = to_filtered(data)
    x, tag = to_morphs(x, includes=['NNG', 'NNP', 'VV'])
    x, tag = select_hangul(x, tag)
    assert x.shape == (len(data),), x.shape

    x, tag = to_w2v_texts(w2v, x, tag)
    return x, tag


def print_cluster_labels(labels, data):
    clusters = defaultdict(list)
    for i, label in enumerate(tuple(labels)):
        if label >= 0:
            clusters[label].append(i)

    for label, indexes in clusters.items():
        for i in indexes:
            title = data.ix[i, 4]
            print('%d: %s' % (label, title[:100]))


def load_w2v():
    w2vpath = os.path.join(
        flag.workdir, os.pardir, 'word2vec',
        str(flag.w2v_version), 'saved.bin')
    return KeyedVectors.load_word2vec_format(w2vpath).wv


data = read_data()
train = data['content'].astype(str)

w2v = load_w2v()

# preprocessing
x = pd.Series()
for i in range(0, len(train), chunksize):
    chunk = train[i:i+chunksize]
    x_, xt = preproc(chunk, w2v)
    assert x_.shape == (len(chunk),)

    x_ = pd.Series(x_)
    x = pd.concat((x, x_), ignore_index=True)


# fit tfidf tokenizer
tokenizer = Tokenizer(num_words=flag.vocabsize, filters='')
tokenizer.fit_on_texts(x)


scores = pd.DataFrame()
for i in range(0, len(x), chunksize):
    chunk = x[i:i+chunksize]

    score = to_word_vectors(
        w2v, chunk, size=w2v.vector_size, tfidfer=tokenizer)
    assert score.shape == (len(chunk), w2v.vector_size), score.shape

    score = pd.DataFrame(score)
    scores = pd.concat((scores, score), ignore_index=True)


dbscan = DBSCAN(
    eps=flag.eps,
    min_samples=flag.coresize,
    metric='euclidean',
    n_jobs=-1)
labels = dbscan.fit_predict(scores)

print_cluster_labels(labels, data)
