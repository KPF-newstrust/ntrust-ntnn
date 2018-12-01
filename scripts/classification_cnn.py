import argparse
import csv
import os

import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from ntnn.constant import categories, token
from ntnn.util import vectorize_category, vectorize_str
from ntnn.classification_cnn import to_filtered, build_model


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--workdir', default='./.works/classification_cnn')
parser.add_argument('--version', default=1)
parser.add_argument('--seqlen', default=1000)
parser.add_argument('--batchs', default=10)
parser.add_argument('--epochs', default=10)
parser.add_argument('--nrows', default=None, type=int)

flag = parser.parse_args()


def read_data():
    return pd.read_csv(
        os.path.join(flag.workdir, 'train.csv'),
        nrows=flag.nrows,
        header=0,
        delimiter='|',
        skipinitialspace=True,
        quoting=csv.QUOTE_MINIMAL,
        iterator=True,
        chunksize=1000)


def preproc(data):
    label = data['category'].astype(str)
    label = vectorize_category(label, dtype=np.int16)
    label = to_categorical(label, num_classes=len(categories))

    train = data['content'].astype(str)
    train = to_filtered(train)
    train = vectorize_str(train, maxlen=flag.seqlen, dtype=np.int16)
    n_trains = int(train.shape[0] * .8)
    x, y = train[:n_trains], label[:n_trains]
    tx, ty = train[n_trains:], label[n_trains:]
    return x, y, tx, ty


def do_train(model, x, label):
    model.fit(
        x, label,
        batch_size=flag.batchs,
        epochs=flag.epochs,
        validation_split=0.1)


def do_eval(model, x, label):
    loss, acc = model.evaluate(x, label, batch_size=1)
    print('loss / accuracy = {:.4f} / {:.4f}'.format(loss, acc))


def save_model(model):
    outdir = os.path.join(flag.workdir, str(flag.version))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, 'saved.h5')
    model.save(outpath)
    print('model saved to %s' % outpath)


data = read_data()

model = build_model(
    token.Total, len(categories), seqlen=flag.seqlen)
model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

for chunk in data:
    x, y, tx, ty = preproc(chunk)

    do_train(model, x, y)

    do_eval(model, tx, ty)

save_model(model)
