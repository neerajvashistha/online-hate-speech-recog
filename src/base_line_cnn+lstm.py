import nltk
# from hate_speech import HateSpeech

import os
import re
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib
from sklearn import preprocessing
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pdb

class HateSpeech:

    TAG_RE = re.compile(r'<[^>]+>')
    EPOCHS = 10
    BATCH_SIZE = 30
    SEED = 8
    INPUT_PATH = os.path.join(os.path.dirname(__file__), 'input/') if os.environ.get('EXECUTION_ENV') is None else '/input'
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'output/') if os.environ.get('EXECUTION_ENV') is None else '/output'

    def __init__(self, load_weights=False, training=False):
        '''

        HateSpeech model returns binary classification predictions for tweet containing hate speech or not.

        Keyword arguments:

        load_weights -- boolean

            If a instance is created with load_weights == True, then the instance will load trained weights from weights.best.hdf5 and use those for the network.

        training -- boolean

            Pass True if you wish to train new weights for this model.

        '''

        self.training = training

        if training:
            X, self.Y = self._get_data()
            documents = self._clean_documents(X[:,0])
            vocabulary = self._get_vocabulary(documents)
            vocab_size = len(vocabulary) + 1
            tokenizer = self._get_tokenizer(documents)
            max_document_length = max([len(s.split()) for s in documents])
            encoded = tokenizer.texts_to_sequences(documents)
            self.X = sequence.pad_sequences(encoded, maxlen=max_document_length)
            self.model = self._get_model(vocab_size, max_document_length)
            #pdb.set_trace()
            print("Vocabulary size: %s" % vocab_size)
            self.train()
        else:
            self.model = self._get_model(load_weights=load_weights)

    def predict(self, X=None):
        '''

        Return predictions based on model.

        Returns:

        Numpy array for predictions (or None), message


        '''
        if self.training: return None, "Cannot predict when in training!"

        try:
            X = np.reshape(X, (1,8)) # ensure it is the right shape for predicting
            classes = self.model.predict_classes(X, batch_size=self.BATCH_SIZE)
            return classes, "Success!"
        except ValueError as error:
            return None, error

    def train(self):
        '''

        Train a new set of weights.

        Accepts no arguments. For tuning, please see class constants at top of the file. Class must be instanciated in training mode for training to be possible.

        '''

        if self.training:
            np.random.seed(self.SEED)
            weights_path = os.path.join(self.OUTPUT_PATH, 'weights.best.hdf5')
            checkpoint = ModelCheckpoint(weights_path, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            self.model.fit(self.X, self.Y, validation_split=0.2, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=1, callbacks=callbacks_list)
        else:
            print("You did not instanciate this class for training!")


    # INTERNAL METHODS BELOW

    def _get_vocabulary(self, documents):
        return set(" ".join(documents).split())

    def _get_tokenizer(self, documents):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(documents)
        return tokenizer

    def _clean_documents(self, documents):
        # convert to list
        documents = documents.tolist()
        # clean docs
        print("Starting cleaning %s documents" % len(documents))
        documents = [self._clean_doc(doc) for doc in documents]
        return documents

    def _clean_doc(self, doc):
        # remove HTML tags
        doc = self._clean_tags(doc)
        # replace all newlines and tabs
        doc = doc.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
        # add missing space after full stops and commas
        doc = re.sub(r'(?<=[.,])(?=[^\s])', r' ', doc)
        # create tokens
        tokens = word_tokenize(doc)
        # downcase
        tokens = [w.lower() for w in tokens]
        # regexp
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        stripped = [re_punc.sub('', w) for w in tokens]
        # remove non-alphabetic tokens
        words = [word for word in stripped if word.isalpha()]
        # remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        # reduce word to base
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        print('.', end='', flush=True)
        return " ".join(stemmed)

    def _clean_tags(self, text):
        return self.TAG_RE.sub('', text)

    def _get_model(self, vocabulary_size=1, input_length=1, load_weights=False):
        try:
            model = Sequential()
            model.add(Embedding(vocabulary_size, 16, input_length=input_length))
            model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
            model.add(Dropout(0.7))
            model.add(MaxPooling1D(5))
            model.add(LSTM(10))
            model.add(Dense(1, activation='sigmoid'))
            if load_weights:
                weights_path = os.path.join(self.OUTPUT_PATH, 'weights.best.hdf5')
                model.load_weights(weights_path)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
            return model
        except ValueError:
            return None

    def _get_data(self):
        try:
            csv_path = os.path.join(self.INPUT_PATH, 'labeled_data.csv')
            X = pd.read_csv(csv_path, sep=',', header=None, error_bad_lines=False) # use pandas to read CSV
            X = X.dropna() # drop any rows with nans

            # pick hateful instances
            hateful = X[5] == 0

            # pick neutral instances
            neutral = X[5] == 2

            # filter
            X = X[hateful | neutral]
            X = np.asarray(X) # convert to array

            # balance data into 50% / 50% according to Y
            idx_0 = np.where(X[:, 5] == 0)[0]
            idx_1 = np.where(X[:, 5] == 2)[0]
            n = min(idx_0.size, idx_1.size)
            idx_d = max([idx_0, idx_1], key=len)
            np.random.shuffle(idx_d)
            idx_d = idx_d[:n]
            np.delete(X, idx_d, 0)

            # split into X and Y
            Y = X[:, 5] # grab last column
            Y[Y == 2] = 1 # swap 2 for 1
            X = X[:, np.r_[6]] # grab 7 first columns
            return X, Y
        except ValueError:
            return None, None


if __name__ == '__main__':
	nltk.download('punkt')
	nltk.download('stopwords')
	model = HateSpeech(training=True)

