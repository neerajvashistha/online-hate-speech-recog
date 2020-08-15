import nltk
# from hate_speech import HateSpeech

import os
import re
import string
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split
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
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slang import slang

class HateSpeech:

    TAG_RE = re.compile(r'<[^>]+>')
    EPOCHS = 10
    BATCH_SIZE = 30
    SEED = 8
    INPUT_PATH = os.path.join(os.path.dirname(__file__), '../data') if os.environ.get('EXECUTION_ENV') is None else './data'
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../model') if os.environ.get('EXECUTION_ENV') is None else '../model'
    porter = PorterStemmer()
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
            'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="twitter", 

        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="twitter", 

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons,slang]
    )

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
            X, Y = self._get_data()
#             pdb.set_trace()
            X = self._clean_documents(X[:,0])
    
            X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2)
            
            vocabulary = self._get_vocabulary(X_train)
            vocab_size = len(vocabulary) + 1
            
            tokenizer = self._get_tokenizer(X_train)
            tokenizer_test = self._get_tokenizer(X_test)
            
            max_document_length = max([len(s.split()) for s in X_train])
            
            encoded = tokenizer.texts_to_sequences(X_train)
            encoded_test = tokenizer_test.texts_to_sequences(X_test)
            
            self.X = sequence.pad_sequences(encoded, maxlen=max_document_length)
            self.Y = y_train
            
            self.X_test = sequence.pad_sequences(encoded_test, maxlen=max_document_length)
            self.Y_test = y_test
#             pdb.set_trace()
            self.model = self._get_model(vocab_size, max_document_length)
            #pdb.set_trace()
            print("Vocabulary size: %s" % vocab_size)
            self.train()
            self.test(vocab_size, max_document_length)

#     def predict(self, X=None):
#         '''

#         Return predictions based on model.

#         Returns:

#         Numpy array for predictions (or None), message


#         '''
#         if self.training: return None, "Cannot predict when in training!"

#         try:
#             X = np.reshape(X, (1,4)) # ensure it is the right shape for predicting
#             classes = self.model.predict_classes(X, batch_size=self.BATCH_SIZE)
#             return classes, "Success!"
#         except ValueError as error:
#             return None, error

    def train(self):
        '''

        Train a new set of weights.

        Accepts no arguments. For tuning, please see class constants at top of the file. Class must be instanciated in training mode for training to be possible.

        '''

        if self.training:
            np.random.seed(self.SEED)
            weights_path = os.path.join(self.OUTPUT_PATH, 'weights.best.hdf5')
            checkpoint = ModelCheckpoint(weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#             checkpoint = ModelCheckpoint(weights_path,monitor='val_loss',mode='min',save_best_only=True,verbose=1)
            earlystop = EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = 3, verbose = 1,restore_best_weights = True)
            callbacks_list = [checkpoint,earlystop]
#             callbacks_list = [checkpoint]
            self.model.fit(self.X, self.Y, validation_split=0.1, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=1, callbacks=callbacks_list)
        else:
            print("You did not instanciate this class for training!")

    def test(self,vocab_size, max_document_length):
        weights_path = os.path.join(self.OUTPUT_PATH, 'weights.best.hdf5')
        self.model = self._get_model(vocab_size, max_document_length, load_weights_path=weights_path)
        scores = self.model.evaluate(self.X_test, self.Y_test, batch_size=self.BATCH_SIZE, verbose=1)
        print(self.model.metrics_names, scores)
        
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
#         pdb.set_trace()
        # clean docs
        print("Starting cleaning %s documents" % len(documents))
        documents = [self._clean_doc_ekphrasis(doc) for doc in documents]
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
        
        stemmed = [self.porter.stem(word) for word in words]
        print('.', end='', flush=True)
        return " ".join(stemmed)

    def _clean_doc_ekphrasis(self, doc):
        words = self.text_processor.pre_process_doc(doc)
        filter_tags = ['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<repeated>','<date>','<elongated>','<url>','<email>','<percent>','<phone>','<date>','<number>']
        stop_words = set(stopwords.words('english')+filter_tags)
#         pdb.set_trace()
        words = [w.replace('<','').replace('>','') for w in words if not w in stop_words and (len(w) > 2 or w in string.punctuation)]
        stemmed = [self.porter.stem(word) for word in words]
        print('.', end='', flush=True)
#         pdb.set_trace()
        return " ".join(stemmed)
        
    
    def _clean_tags(self, text):
        return self.TAG_RE.sub('', text)

    def _get_model(self, vocabulary_size=1, input_length=1, load_weights_path=None):
        try:
            model = Sequential()
            model.add(Embedding(vocabulary_size, 16, input_length=input_length))
            model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
            model.add(Dropout(0.7))
            model.add(MaxPooling1D(5))
            model.add(LSTM(10))
            model.add(Dense(3))
            model.add(Activation('sigmoid'))
            if load_weights_path:
                model.load_weights(load_weights_path)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            print(model.summary())
            return model
        except ValueError:
            return None

    def _get_data(self):
        try:
            csv_path = os.path.join(self.INPUT_PATH, 'dataset.csv')
            X = pd.read_csv(csv_path, sep=',', error_bad_lines=False) # use pandas to read CSV
            X = X.dropna() # drop any rows with nans
            X.drop(X[(X['text_id'] == 'text_id')].index,inplace=True)
            X.drop(X[(X['hate'] == 'HS')].index,inplace=True)
            X["hate"] = X["hate"].astype(str).astype(int)
            X = X.sample(n=len(X), random_state=42)
#             X.rename(columns={'hate':'class'},inplace=True)
#             pdb.set_trace()
            # pick neutral instances
#             neutral = X['class'] == '0'

#             # pick hateful instances
#             hateful = X['class'] == '1'

#             # filter
#             X = X[hateful | neutral]
#             X = np.asarray(X) # convert to array
# #             pdb.set_trace()
#             # balance data into 50% / 50% according to Y
#             idx_0 = np.where(X[:, 2] == '0')[0]
#             idx_1 = np.where(X[:, 2] == '1')[0]
#             n = min(idx_0.size, idx_1.size)
#             idx_d = max([idx_0, idx_1], key=len)
#             np.random.shuffle(idx_d)
#             idx_d = idx_d[:n]
#             np.delete(X, idx_d, 0)

#             # split into X and Y
#             Y = X[:, 2] # grab last column
#             Y[Y == '1'] = 1 
#             Y[Y == '0'] = 0
            
#             X = X[:, np.r_[1]] # grab 7 first columns
            
            x = X[["text"]].to_numpy()
            y = to_categorical(X["hate"].to_numpy())
            return x, y
        except ValueError:
            return None, None


if __name__ == '__main__':
	nltk.download('punkt')
	nltk.download('stopwords')
	model = HateSpeech(load_weights=False,training=True)

