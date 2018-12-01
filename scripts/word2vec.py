import argparse
import csv
import os
import pandas as pd

from gensim.models import Word2Vec
from ntnn.util import to_filtered, to_morphs, to_sentences, EpochLogger
from ntnn.word2vec import eval_w2v, StrCorpus


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--workdir', default='./.works/word2vec')
parser.add_argument('--version', default=1)
parser.add_argument('--vocabsize', default=500000)
parser.add_argument('--batchs', default=500)
parser.add_argument('--epochs', default=3)
parser.add_argument('--nrows', default=None, type=int)

flag = parser.parse_args()
chunksize = 5000


def read_data():
    return pd.read_csv(
        os.path.join(flag.workdir, 'train.csv'),
        nrows=flag.nrows,
        header=0,
        delimiter='|',
        skipinitialspace=True,
        quoting=csv.QUOTE_MINIMAL,
        iterator=True,
        chunksize=chunksize)


def preproc(x):
    x = to_filtered(x)
    x = to_sentences(x)
    assert x.ndim == 1

    x, tags = to_morphs(x)
    return x


def build_and_train_model(train):
    # Dimensionality of the word vectors
    wv_dim = 100

    # Max dist between the current and predicted word within a sentence.
    window = 8

    corpus = StrCorpus(train)
    logger = EpochLogger(flag.epochs)

    return Word2Vec(
        corpus,
        size=wv_dim,
        window=window,
        min_count=3,  # 빈도 1개이상만 처리
        max_vocab_size=flag.vocabsize,
        sg=1,  # 1: skipgram, 0: cbow
        hs=0,
        workers=1,
        batch_words=flag.batchs,
        iter=flag.epochs,
        callbacks=[logger])


def save_model(model):
    outdir = os.path.join(flag.workdir, str(flag.version))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, 'saved.bin')
    model.wv.save_word2vec_format(outpath)
    print('model saved to %s' % outpath)


def do_eval(model):
    print(eval_w2v(model, ['미국', '대통령']))
    print(eval_w2v(model, ['가수', '그룹']))


data = read_data()

train = pd.Series()
for chunk in data:
    chunk = chunk['content'].astype(str)
    chunk_ = preproc(chunk)
    train = pd.concat((train, pd.Series(chunk_)), ignore_index=True)
print('train %d sentences' % len(train))

model = build_and_train_model(train)
print('trained %d words' % len(model.wv.vocab))

do_eval(model)

save_model(model)
