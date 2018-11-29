import os
import csv
import re
import tensorflow as tf
from enum import Enum


class Mode(Enum):
    train=0
    test=1
    predict=2


def shape(tensor):
    return tensor.get_shape().as_list()


def read_random_line(fd):
    fd.seek(0, os.SEEK_END)
    total_bytes = fd.tell()  #os.stat(file_name).st_size 

    random_point = random.randint(0, total_bytes)

    fd.seek(random_point)
    try: 
        fd.readline() # skip this line to clear the partial line
    except: pass

    line = fd.readline()
    if len(line) == 0:
        f.seek(0)
        return fd.readline()
    else:
        return line


def line_to_word_ids(lines):
    line_ids = []
    for line in lines:
        line = line.decode('utf-8')
        ids = []
        for word in RE.findall(line)[:max_doc]:
            ids.append(word_ids[word] if word in word_ids else len(word_ids))

        ids += [len(word_ids)] * (max_doc - len(ids))
        line_ids.append(ids)
    return tf.convert_to_tensor(line_ids, dtype=tf.int32)
    

def read_csv(csv_file):
    features, labels = [], []
    with open(csv_file, encoding='utf8') as f:
        reader = csv.reader(f, delimiter='|', escapechar=':', quoting=csv.QUOTE_NONE, skipinitialspace=True)
        for row in reader:
            if len(row) != 2: continue
            try:
                labels.append(int(row[0]))
                features.append(row[1])
            except:
                pass

    print('--- n of docs', len(features))
    return features, labels


def _tokenizer(iterator):
    RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]{2,}(?=[A-Z])|[\'\w\-]{2,}", re.UNICODE)
    for val in iterator:
        yield RE.findall(val)

def load_vocab(db_dir, max_doc=50, tokenizer=None, min_frequency=0):
    vocab_data_file = os.path.join(db_dir, 'vocab.db')
    if os.path.exists(vocab_data_file):
        vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_data_file)
        vp.vocabulary_.freeze(False)
    else:
        tok = tokenizer or _tokenizer
        vp = tf.contrib.learn.preprocessing.VocabularyProcessor(max_doc, tokenizer_fn=tok, min_frequency=min_frequency)
    print('load vocabularies..', len(vp.vocabulary_))
    return vp


def save_vocab(vocab_proc, db_dir):
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    vocab_data_file = os.path.join(db_dir, 'vocab.db')
    vocab_proc.save(vocab_data_file)
    print('save vocabularies..', vocab_data_file, len(vocab_proc.vocabulary_))


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def clear_str(string):
    string = re.sub(r"[^A-Za-z0-9\uAC00-\uD7AF(),!?\'\`]", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


