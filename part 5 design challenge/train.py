## LOAD GLOVE INTO EMBEDDING MATRIX
import os
import numpy as np

GLOVE_DIR = "glove_path"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

## train model
import tensorflow as tf
import pandas as pd
import os
import sys

################ EDIT DATA FOLDER AND LANGUAGE HERE ACCORDINGLY ################
main_path = 'data'
train_lang = 'EN'  ## ES,EN
################################################################################

data_path = os.path.join(os.getcwd(), '{}/{}/train'.format(main_path, train_lang))

sent_index = []
text = []
label = []
sent_count = 1
prev = None
with open(data_path, encoding='utf-8', mode='r') as f:
    for line in f.readlines():

        if line == '\n':
            sent_count += 1
            continue
        if line == '\n' and prev == '\n':
            break
        sent_index.append(sent_count)
        t, l = line.strip().split()
        text.append(t)
        label.append(l)
        prev = line

ner_data = pd.DataFrame(columns=['sentence_idx', 'word', 'tag'])
ner_data['sentence_idx'] = sent_index
ner_data['word'] = text
ner_data['tag'] = label


class SentenceGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                     s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(ner_data)
sentences = getter.sentences

maxlen = max([len(s) for s in sentences])
print('Maximum sequence length:', maxlen)

words = list(set(ner_data["word"].values))
words.append("ENDPAD")
words.append("NA")
tags = list(set(ner_data["tag"].values))
n_tags = len(tags)
n_words = len(words)

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

EMBEDDING_DIM = 300
embedding_matrix = np.random.random((len(word2idx) + 1, EMBEDDING_DIM))
for word, i in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# embedding_layer = Embedding(len(word2idx) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=maxlen,
#                             trainable=True,
#                             mask_zero=True)
idx2word = {v: k for k, v in word2idx.items()}
idx2tag = {v: k for k, v in tag2idx.items()}

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

X = [[word2idx[w[0]] for w in s] for s in sentences]

X = pad_sequences(maxlen=maxlen, sequences=X, padding="post", value=word2idx['ENDPAD'])
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=maxlen, sequences=y, padding="post", value=tag2idx["O"])
from keras.utils import to_categorical

y = [to_categorical(i, num_classes=n_tags) for i in y]
from sklearn.model_selection import train_test_split

X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2)

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

input = Input(shape=(maxlen,))
# model = Embedding(input_dim=n_words, output_dim=maxlen)(input)
model = Embedding(len(word2idx) + 1,
                  EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=maxlen,
                  trainable=True,
                  mask_zero=True)(input)
model = Dropout(0.12)(model)
model = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.1))(model)
# model = Bidirectional(LSTM(units=16, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

model = Model(input, out)

print(model.summary())

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1,
                              min_lr=0.00001)
checkpoint = ModelCheckpoint('{}_ner_model.h5'.format(train_lang), monitor='val_loss',
                             verbose=2, save_best_only=True,
                             save_weights_only=False, mode='min')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3,
                           verbose=2, mode='min')

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, np.array(y_train), batch_size=16, epochs=30,
                    validation_split=0.2, verbose=1, validation_data=(X_dev, np.array(y_dev)),
                    callbacks=[reduce_lr, checkpoint, early_stop])

print('saving dictionairies')
import pickle

with open('word2idx_{}.pickle'.format(train_lang), 'wb') as handle:
    pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tag2idx_{}.pickle'.format(train_lang), 'wb') as handle:
    pickle.dump(tag2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('training finished ')