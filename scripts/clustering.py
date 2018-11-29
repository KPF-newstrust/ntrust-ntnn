import argparse
import csv
import os
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.cluster import DBSCAN
from keras.preprocessing.text import Tokenizer
from ntnn.util import to_morphs
from ntnn.clustering import to_filtered


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('-workdir', default='./.works/clustering')
parser.add_argument('-version', default=1)
parser.add_argument('-vocabsize', default=2000)
parser.add_argument(
    '-eps', default=10, help='neighborhood로 인정되는 최대 거리')
parser.add_argument('-coresize', default=3)

flag = parser.parse_args()
nrows = 500


# Read data
data = pd.read_csv(
    os.path.join(flag.workdir, 'train.csv'),
    nrows=nrows,
    header=0,
    delimiter='|',
    skipinitialspace=True,
    quoting=csv.QUOTE_NONE)
train = data.ix[:, 3].values.astype('str')


# Preprocessing
train = to_filtered(train)
train, tag = to_morphs(train, includes=['NNG', 'NNP', 'VV'])
assert train.shape == (nrows,)

tokenizer = Tokenizer(num_words=flag.vocabsize, filters='')
tokenizer.fit_on_texts(train)
tfidf = tokenizer.texts_to_matrix(train, mode='tfidf')
assert tfidf.shape == (nrows, flag.vocabsize)


# Clustering
dbscan = DBSCAN(
    eps=flag.eps,
    min_samples=flag.coresize,
    metric='euclidean',
    n_jobs=-1)
labels = dbscan.fit_predict(tfidf)


# Grouping
clusters = defaultdict(list)
for i, label in enumerate(tuple(labels)):
    if label >= 0:
        clusters[label].append(i)

for label, indexes in clusters.items():
    for i in indexes:
        title = data.ix[i, 4]
        print('%d: %s' % (label, title[:100]))
