import nltk
import os
import re
import string
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
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

class CNN_LSTM:
	def __init__(self, seed, epochs, batch_size):
		# seed = 8
		# epochs = 10
		# batch_size = 30
		self.seed = seed
		self.epochs = epochs
		self.batch_size = batch_size


	def get_model(self, classes, vocabulary_size=1, input_length=1):
		try:
			model = Sequential()
			model.add(Embedding(vocabulary_size, 16, input_length=input_length))
			model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
			model.add(Dropout(0.7))
			model.add(MaxPooling1D(5))
			model.add(LSTM(10))
			model.add(Dense(classes))
			model.add(Activation('sigmoid'))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			print(model.summary())
			return model
		except ValueError:
			return None


	def train(self, X, y, model, weights_path):

		np.random.seed(self.seed)
		checkpoint = ModelCheckpoint(weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
		# checkpoint = ModelCheckpoint(weights_path,monitor='val_loss',mode='min',save_best_only=True,verbose=1)
		earlystop = EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = 3, verbose = 1,restore_best_weights = True)
		callbacks_list = [checkpoint,earlystop]
		
		return model.fit(X, y, validation_split=0.1, epochs=self.epochs, batch_size=self.batch_size, verbose=1, callbacks=callbacks_list)

	def test(self,model,X_test, Y_test):
		scores = model.evaluate(X_test, Y_test, batch_size=self.batch_size, verbose=1)
		return scores

	def predict(self,model_path,x):
		mod = load_model(model_path)
		return mod.predict(x)


class ProcessData:

	def __init__(self, df):

		self.tokenizer = Tokenizer()
		self.max_seq_len = None
		self.vocab_size = None
		self.classes = None

		X_train, X_test, y_train, y_test= map(self.get_data, [df])
		self.X_train, self.X_test, self.y_train, self.y_test = convert_text_to_examples(self, X_train, X_test, y_train, y_test)
		

	def convert_text_to_examples(self, X_train, X_test, y_train, y_test):

		vocabulary = set(" ".join(X_train).split())
		self.vocab_size = len(vocabulary) + 1

		tokenizer = self.tokenizer.fit_on_texts(X_train)
		tokenizer_test = self.tokenizer.fit_on_texts(X_test)

		self.max_seq_len = max([len(s.split()) for s in X_train])

		encoded = tokenizer.texts_to_sequences(X_train)
		encoded_test = tokenizer_test.texts_to_sequences(X_test)

		X = sequence.pad_sequences(encoded, maxlen=self.max_seq_len)
		Y = y_train

		X_test = sequence.pad_sequences(encoded_test, maxlen=self.max_seq_len)
		Y_test = y_test
		return X, X_test, Y, Y_test


	def get_data(self, df):
		X = df.dropna() # drop any rows with nans
		X.drop(X[(X['text_id'] == 'text_id')].index,inplace=True)
		X.drop(X[(X['hate'] == 'HS')].index,inplace=True)
		X["hate"] = X["hate"].astype(str).astype(int)
		X = X.sample(n=len(X), random_state=42)		
		x = X[["text"]].to_numpy()
		self.classes = X.groupby(["hate"]).count()
		y = to_categorical(X["hate"].to_numpy())
		x = _clean_doc(x)
		X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
		return X_train, X_test, y_train, y_test


	def clean_doc(self, documents):
		documents = documents.tolist()
#         pdb.set_trace()
        # clean docs
        clean_doc_list = []
        print("Starting cleaning %s documents" % len(documents))
        for doc in tqdm(documents):
			# remove HTML tags
			doc = re.compile(r'<[^>]+>').sub('', text)
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
			clean_doc_list.append(" ".join(stemmed))
		return clean_doc_list