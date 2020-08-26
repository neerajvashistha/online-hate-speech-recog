import os, re, string
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Activation, Embedding, Reshape, concatenate, Conv1D, Conv2D
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, Flatten, MaxPooling1D, Embedding
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np, pdb, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm 
import matplotlib.pyplot as plt
from utils.hindi_tokenizer import HindiTokenizer
from inltk.inltk import tokenize as hi_tokenizer
import feature_generation as fg
from sklearn.utils.class_weight import compute_class_weight

class CNN_LSTM:
    def __init__(self, param = None):
        # seed = 8
        # epochs = 10
        # batch_size = 30
        self.seed = param['seed']
        self.epochs = param['epochs']
        self.batch_size = param['batch_size']
        self.optimiser = param['optimiser']
        self.lr_rate = param['lr_rate']
        self.val_split=param['val_split']


    def get_model(self, classes, vocabulary_size=1, input_length=1,param=None):
        try:
            if self.optimiser == 'sgd':
                opt = SGD(lr=self.lr_rate)
            if self.optimiser == 'adam':
                opt = Adam(lr=self.lr_rate)
            drop = param['drop_out']
            HIDDEN_SIZE = param['hidden_size']
            model = Sequential()
            model.add(Embedding(vocabulary_size, 16, input_length=input_length))
            model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
            model.add(Dropout(0.7))
            model.add(MaxPooling1D(5))
            model.add(LSTM(10))
            model.add(Dense(classes))
            model.add(Activation('sigmoid'))
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            print(model.summary())
            return model
        except ValueError:
            return None
    def get_model_2(self, classes, vocabulary_size=1, input_length=1, param=None):
        try:
            filter_sizes = [3,4,5]
            num_filters = 64
            drop = param['drop_out']
            VOCAB_SIZE = vocabulary_size # len(wordvectors) # 43,731
            MAX_LENGTH = input_length # len(max(sentences, key=len))
            EMBED_SIZE = 100 # arbitary
            HIDDEN_SIZE = param['hidden_size'] # len(unique_tags)
            
            if self.optimiser == 'sgd':
                opt = SGD(lr=self.lr_rate)
            if self.optimiser == 'adam':
                opt = Adam(lr=self.lr_rate)
            
            # CNN model
            inputs = Input(shape=(MAX_LENGTH, ), dtype='int32')
            embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE, input_length=MAX_LENGTH)(inputs)
            reshape = Reshape((MAX_LENGTH, EMBED_SIZE, 1))(embedding)

            # 3 convolutions
            conv_0 = Conv2D(num_filters, 
                            kernel_size=(filter_sizes[0], EMBED_SIZE), 
                            strides=1, padding='valid', 
                            kernel_initializer='normal', 
                            activation='relu')(reshape)
            bn_0 = BatchNormalization()(conv_0)
            conv_1 = Conv2D(num_filters, 
                            kernel_size=(filter_sizes[1], EMBED_SIZE), 
                            strides=1, padding='valid', 
                            kernel_initializer='normal', 
                            activation='relu')(reshape)
            bn_1 = BatchNormalization()(conv_1)
            conv_2 = Conv2D(num_filters, 
                            kernel_size=(filter_sizes[2], EMBED_SIZE), 
                            strides=1, padding='valid', 
                            kernel_initializer='normal', 
                            activation='relu')(reshape)
            bn_2 = BatchNormalization()(conv_2)

            # maxpool for 3 layers
            maxpool_0 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[0] + 1, 1), padding='valid')(bn_0)
            maxpool_1 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[1] + 1, 1), padding='valid')(bn_1)
            maxpool_2 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[2] + 1, 1), padding='valid')(bn_2)

            # concatenate tensors
            concatenate_tensors = concatenate([maxpool_0, maxpool_1,maxpool_2],axis=-1)
            # flatten concatenated tensors
            flatten_concat = Flatten()(concatenate_tensors) # dim 2
            # dense layer (dense_1)
            dense_1 = Dense(units=HIDDEN_SIZE,input_shape=(1,),activation='relu')(flatten_concat) # dim 2
            # dropout_1
            dropout_1 = Dropout(drop)(dense_1) # dim 2
            
            # BLSTM model
            # time_xx = TimeDistributed(Flatten())(concatenate_tensors)
            time_xx = TimeDistributed(Dense(HIDDEN_SIZE))(concatenate_tensors)

            _CNN_to_LSTM = Reshape((time_xx.shape[1]*time_xx.shape[2],
                                    int(time_xx.shape[3])),
                                   input_shape=time_xx.shape[1:4])(time_xx)
            # https://github.com/keras-team/keras/issues/11425
            # _CNN_to_LSTM = Dense(units=HIDDEN_SIZE,input_shape=(1,),activation='relu')(time_xx)
            # _CNN_to_LSTM = Dropout(drop)(_CNN_to_LSTM)

            # Bidirectional 1
            b1 = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))(_CNN_to_LSTM)
            # Bidirectional 2
            b2 = Bidirectional(LSTM(HIDDEN_SIZE))(b1)
            # Dense layer (dense_2)
            dense_2 = Dense(HIDDEN_SIZE)(b2)
            # dropout_2
            dropout_2 = Dropout(drop)(dense_2)           
            

            # concatenate 2 final layers
            y = concatenate([dropout_1, dropout_2],axis = -1)
            # output
            # out = Activation('sigmoid')(y)
            out = Dense(classes)(y)
            out = Activation('sigmoid')(out)
            m = Model(inputs,out)
            m.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'], )
            print(m.summary())
            return m
        except ValueError:
            return None

    def train(self, X, y, model, weights_path, d_class_weights):

        np.random.seed(self.seed)
        checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # checkpoint = ModelCheckpoint(weights_path,monitor='val_loss',mode='min',save_best_only=True,verbose=1)
        earlystop = EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = 5, verbose = 1,restore_best_weights = True)
        callbacks_list = [checkpoint,earlystop]
        
        return model.fit(X, y, validation_split=self.val_split, 
                         epochs=self.epochs, 
                         batch_size=self.batch_size, 
                         verbose=1, 
                         class_weight=d_class_weights,
                         callbacks=callbacks_list)

    def test(self,model,X_test, Y_test):
        scores = model.evaluate(X_test, Y_test, batch_size=self.batch_size, verbose=1)
        return scores

    def predict(self,model_path,x):
        mod = load_model(model_path)
        return [mod.predict_classes(x),mod.predict(x)]
    
    def plot_taining_graphs(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        


class ProcessData:

    def __init__(self, df,lang,max_seq_len=None):

        self.porter = PorterStemmer()
        self.max_seq_len = max_seq_len
        self.vocab_size = None
        self.classes = None
        self.lang = lang
        self.df = df
        
        if isinstance(self.df, list):
        	self.X = self.list_to_examples(self.df)
        else:
	        (X_train, X_test, y_train, y_test) = self.get_data(self.df)
	        self.X_train, self.X_test, self.y_train, self.y_test, self.class_weight = self.convert_text_to_examples(X_train, X_test, y_train, y_test)


    def convert_text_to_examples(self, X_train, X_test, y_train, y_test):

        vocabulary = set(" ".join(X_train).split())
        self.vocab_size = len(vocabulary) + 1

        tokenizer = self.get_tokenizer(X_train)
        tokenizer_test = self.get_tokenizer(X_test)
#         pdb.set_trace()
        self.max_seq_len = max([len(s.split()) for s in X_train])
        # a = {s:len(s.split()) for s in X_train}
        # v = [i for i in list(a.keys()) if a[i]==self.max_seq_len]
        encoded = tokenizer.texts_to_sequences(X_train)
        encoded_test = tokenizer_test.texts_to_sequences(X_test)

        X = sequence.pad_sequences(encoded, maxlen=self.max_seq_len)
        Y = y_train

        X_test = sequence.pad_sequences(encoded_test, maxlen=self.max_seq_len)
        Y_test = y_test
        y_integers = np.argmax(y_test, axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = dict(enumerate(class_weights))
        return X, X_test, Y, Y_test, d_class_weights


    def get_data(self, df):
#         print(type(df))
        df = df.sample(n=len(df), random_state=42)
        self.classes = df["class"].unique()
        
        x = df["text"].to_numpy()        
        y = to_categorical(df["class"].to_numpy())
        x = self.clean_doc(x)
        X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
        return X_train, X_test, y_train, y_test

    def clean_doc(self,documents):
        documents = documents.tolist()
        clean_doc_list = []
        pdb.set_trace()
        feat = fg.features(lang=self.lang)
        for doc in tqdm(documents):
            tweet = feat.preprocess(doc)
            token = feat.tokenize(tweet)
            if self.lang=='en':
                pass
            if self.lang=='hi':
                token = re.sub('[A-Za-z0-9-()\"#/@;:<>{}`+=~|.!?,]+', '', " ".join(token)).split(" ") 
            token = list(filter(str.strip, token))
            clean_doc_list.append(" ".join(token))
        return clean_doc_list
    
#     def clean_doc_old(self, documents):
#         ht = HindiTokenizer.Tokenizer()
#         documents = documents.tolist()
#         pdb.set_trace()
#         # clean docs
#         clean_doc_list = []
# #         print("Starting cleaning %s documents" % len(documents))
#         for doc in tqdm(documents):
#             # remove HTML tags
#             doc = re.compile(r'<[^>]+>').sub('', doc)
#             # replace all newlines and tabs
#             doc = doc.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ').replace('_',' ').replace('।', '').replace('|','')
#             # add missing space after full stops and commas
#             doc = re.sub(r'(?<=[.,])(?=[^\s])', r' ', doc)
#             # create tokens
#             if self.lang == 'en':                
#                 tokens = word_tokenize(doc)
#             if self.lang == 'hi':
# #                 tokens = [w.replace('▁','').replace('।', '').replace('|','') for w in hi_tokenizer(input=doc , language_code=self.lang)]
#                 tokens = word_tokenize(doc)
#             # downcase
#             tokens = [w.lower() for w in tokens]
#             # regexp
#             re_punc = re.compile('[%s]' % re.escape(string.punctuation))            
#             stripped = [re_punc.sub('', w) for w in tokens]
#             # remove non-alphabetic tokens
#             if self.lang == 'en':
#                 words = [word for word in stripped if word.isalpha()]
#             if self.lang == 'hi':
#                 words = stripped
#             # remove stopwords
#             if self.lang == 'en':
#                 stop_words = set(stopwords.words('english'))
#                 words = [w for w in words if not w in stop_words]
#                 # reduce word to base
#                 stemmed = [self.porter.stem(word) for word in words]
#             if self.lang == 'hi':
#                 stop_words = ht.stopwords()
#                 words = [w for w in words if not w in stop_words]
#                 stemmed = [ht.generate_stem_words(w) for w in words]
#             stemmed = list(filter(str.strip, stemmed))
#             clean_doc_list.append(" ".join(stemmed))
#         return clean_doc_list
    
    def get_tokenizer(self, documents):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(documents)
        return tokenizer

    def list_to_examples(self, X_list):
    	x = np.array(X_list) 
    	x = self.clean_doc(x)
    	tokenizer = self.get_tokenizer(x)
    	encoded = tokenizer.texts_to_sequences(x)
    	X = sequence.pad_sequences(encoded, maxlen=self.max_seq_len)
    	return X




