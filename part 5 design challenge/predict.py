import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class SentenceGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [w for w in s["word"].values.tolist()]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


import pickle
import sys

################ EDIT DATA FOLDER AND LANGUAGE HERE ACCORDINGLY ################
main_path = 'test'  ## if you want to write to test folder
test_lang = 'EN'  ## just set this to ES or EN
kind = 'test'  ## generate [kind].p5.out
################################################################################


## code below reads the prediction dataset from respective path defined above,
## load the previous dictionaries generated after running training script

maxlen = 72
if test_lang == 'ES':
    maxlen = 187

with open('word2idx_{}.pickle'.format(test_lang), 'rb') as f:
    word2idx = pickle.Unpickler(f).load()

with open('tag2idx_{}.pickle'.format(test_lang), 'rb') as f:
    tag2idx = pickle.load(f)
print('Dictionairies loaded')

idx2word = {v: k for k, v in word2idx.items()}
idx2tag = {v: k for k, v in tag2idx.items()}

from keras.models import load_model

loaded_model = load_model('{}_ner_model.h5'.format(test_lang))
print('Prediction model loaded')
data_path = '{}/{}/{}.in'.format(main_path, test_lang, kind)
print('Data Path is : ', data_path)


## the function, takes in path where file whose prediction is to be generated is located.
## then, we need to convert those raw text into sequence of numbers of list of list

def load_data(data_path):
    sent_index = []
    text = []
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
            t = line.strip()
            text.append(t)
            prev = line
    d = pd.DataFrame(columns=['sentence_idx', 'word'])
    d['sentence_idx'] = sent_index
    d['word'] = text
    getter = SentenceGetter(d)
    sentences = getter.sentences
    X = [[word2idx.get(w, word2idx['NA']) for w in s] for s in sentences]

    X = pad_sequences(maxlen=maxlen, sequences=X, padding="post", value=word2idx['ENDPAD'])

    return X, sentences


import numpy as np

## X_test is the numerical form of dataset present in data_path
X_test, data_test = load_data(data_path)

## loading the trained model and prediction of values

p = loaded_model.predict(X_test)

## section of code below, takes in prediction p, and writes output, as ***.p1.out file
## which can later be used by evalScripy.py file
y_pred = []
print('Writing to : ', data_path)
for index, x_val in enumerate(X_test):

    preds_val = [idx2tag[j] for j in np.argmax(p[index], axis=-1)]
    for i in range(len(data_test[index])):

        with open('./{}/{}/{}.p5.out'.format(main_path, test_lang, kind), mode='a', encoding='utf-8') as f:
            # print(data_test[index][i],' ',preds_val[i])

            try:
                f.write(data_test[index][i] + ' ' + preds_val[i] + '\n')
                y_pred.append(tag2idx[preds_val[i]])
            except:
                f.write(data_test[index][i] + ' ' + 'O' + '\n')
                y_pred.append(tag2idx['O'])
    with open('./{}/{}/{}.p5.out'.format(main_path, test_lang, kind), 'a') as f:
        f.write('\n')
    # print('\n')