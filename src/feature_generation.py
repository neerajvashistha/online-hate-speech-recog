from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import pandas as pd
import re,numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
import pdb
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slang import slang
from indicnlp.syllable import  syllabifier
from fastai.text import *
from utils.hindi_tokenizer import HindiTokenizer
from inltk.inltk import tokenize as hi_tokenizer
from indicnlp import common
from indicnlp import loader
from nltk.tag import tnt
from nltk.corpus import indian

class features():

    def __init__(self,lang='en'):
        self.lang = lang
        self.stopwords = None
        self.stemmer = None
        self.sentiment_analyzer = None
        self.text_processor = None        
        INDIC_NLP_RESOURCES=r"/home/nv/project/online-hate-speech-recog/model/indic_nlp_resources/"        
        common.set_resources_path(INDIC_NLP_RESOURCES)
        self.pos_tagger = None



        if lang == 'hi':
            self.ht = HindiTokenizer.Tokenizer()
            self.sentiment_analyzer = load_learner(path="../model/hi-sentiment")
            self.stopwords = self.ht.stopwords()
            other_exclusions = ["#ff", "ff", "rt"]
            self.stopwords.extend(other_exclusions)
            self.stemmer = None
            self.text_processor = TextPreProcessor(
                # terms that will be normalized
                normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                    'time', 'url', 'date', 'number'],
                # terms that will be annotated
                annotate={"hashtag", "allcaps", "elongated", "repeated",
                    'emphasis', 'censored'},
                fix_html=True,  # fix HTML tokens
            )
            loader.load()
            train_data = indian.tagged_sents('hindi.pos')
            self.tnt_pos_tagger = tnt.TnT()
            self.tnt_pos_tagger.train(train_data)

        if lang == 'en':
            self.sentiment_analyzer = VS()
            self.stopwords = nltk.corpus.stopwords.words("english")
            other_exclusions = ["#ff", "ff", "rt"]
            self.stopwords.extend(other_exclusions)
            self.stemmer = PorterStemmer()
            self.text_processor = TextPreProcessor(
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


    def preprocess(self,text_string):
        """
        Accepts a text string and replaces:
        1) urls with URLHERE
        2) lots of whitespace with one instance
        3) mentions with MENTIONHERE

        This allows us to get standardized counts of urls and mentions
        Without caring about specific people mentioned
        """
        def dict_replace(wordlist, _dict):
            return [_dict[w] if w in _dict else w for w in wordlist]

        stp_tags = ['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<repeated>','<date>','<elongated>','<url>','<email>','<percent>','<phone>','<date>','<number>']
        if self.lang == 'en':
            words = self.text_processor.pre_process_doc(text_string)
        if self.lang == 'hi':
            sent = self.text_processor.pre_process_doc(text_string)
            words = hi_tokenizer(input=sent , language_code=self.lang)
            for d in [emoticons,slang]:
                    words = dict_replace(words, d)

        words = [w.replace('<','').replace('>','').replace('▁','').replace('।', '').replace('|','') for w in words if not w in stp_tags and (len(w) > 2 or w in string.punctuation)]
        text_string = " ".join(words)
        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@ [\w\-]+'
        parsed_text = re.sub(mention_regex, '', text_string)
        parsed_text = re.sub(space_pattern, ' ', parsed_text)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)
#         pdb.set_trace()
        return parsed_text

    def tokenize(self,tweet):
        """Removes punctuation & excess whitespace, sets to lowercase,
        and stems tweets. Returns a list of stemmed tokens."""
        if self.lang=='en':
            tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
            tokens = [self.stemmer.stem(t) for t in tweet.split()]
            return tokens
        if self.lang=='hi':
            words = [w.replace('▁','').replace('।', '').replace('|','') for w in hi_tokenizer(input=tweet , language_code=self.lang) if w not in self.stopwords]
            tokens = [self.ht.generate_stem_words(w) for w in words]
            return tokens

    def basic_tokenize(self,tweet):
        """Same as tokenize but without the stemming"""
        if self.lang =='en':
            tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
            return tweet.split()
        if self.lang =='hi':
            words = [w.replace('▁','') for w in hi_tokenizer(input=tweet , language_code=self.lang)]
            return words


    def get_tfidf(self,tweets):
        vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize,
            preprocessor=self.preprocess,
            ngram_range=(1, 3),
            stop_words=self.stopwords if self.lang == 'en' else None,
            use_idf=True,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=10000,
            # min_df=5,
            # max_df=0.75,
            )

        #Construct tfidf matrix and get relevant scores
        tfidf = vectorizer.fit_transform(tweets).toarray()
        vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
        idf_vals = vectorizer.idf_
        idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores
        return tfidf,idf_dict,vocab

    def get_pos(self,tweets):
        #Get POS tags for tweets and save as a string
        tweet_tags = []
        if self.lang == 'en':
            for t in tweets:
                tokens = self.basic_tokenize(self.preprocess(t))
                tags = nltk.pos_tag(tokens)
                tag_list = [x[1] for x in tags]
                tag_str = " ".join(tag_list)
                tweet_tags.append(tag_str)
        if self.lang == 'hi':
            for t in tweets:
                tags = self.tnt_pos_tagger.tag(nltk.word_tokenize(t))
                tag_list = [x[1] for x in tags]
                tag_str = " ".join(tag_list)
                tweet_tags.append(tag_str)

        # pdb.set_trace()

        #We can use the TFIDF vectorizer to get a token matrix for the POS tags
        pos_vectorizer = TfidfVectorizer(
            tokenizer=None,
            lowercase=False,
            preprocessor=None,
            ngram_range=(1, 3),
            stop_words=None,
            use_idf=False,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=5000,
            # min_df=5,
            # max_df=0.75,
            )

        #Construct POS TF matrix and get vocab dict
        pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
        pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}
        return pos,pos_vocab



    def count_twitter_objs(self,text_string):
        """
        Accepts a text string and replaces:
        1) urls with URLHERE
        2) lots of whitespace with one instance
        3) mentions with MENTIONHERE
        4) hashtags with HASHTAGHERE

        This allows us to get standardized counts of urls and mentions
        Without caring about specific people mentioned.

        Returns counts of urls, mentions, and hashtags.
        """
        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@[\w\-]+'
        hashtag_regex = '#[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
        parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
        parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
        return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

    def other_features(self,tweet):
        """This function takes a string and returns a list of features.
        These include Sentiment scores, Text and Readability scores,
        as well as Twitter specific features"""

        if self.lang == 'en':
            sentiment = self.sentiment_analyzer.polarity_scores(tweet)
            words = self.preprocess(tweet) #Get text only
            # pdb.set_trace()
            syllables = textstat.syllable_count(words)
            num_chars = sum(len(w) for w in words)
            num_chars_total = len(tweet)
            num_terms = len(tweet.split())
            num_words = len(words.split())
            avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
            num_unique_terms = len(set(words.split()))

            ###Modified FK grade, where avg words per sentence is just num words/1
            FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
            ##Modified FRE score, where sentence fixed to 1
            FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

            twitter_objs = self.count_twitter_objs(tweet)
            retweet = 0
            if "rt" in words:
                retweet = 1
            features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                        num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                        twitter_objs[2], twitter_objs[1],
                        twitter_objs[0], retweet]
            #features = pandas.DataFrame(features)
            return features
        if self.lang == 'hi':
            sentiment = self.sentiment_analyzer.predict(tweet)
            words = self.preprocess(tweet)
            
            syllables = len([syllabifier.orthographic_syllabify(w,self.lang) for w in hi_tokenizer(input=words , language_code=self.lang)])
            # pdb.set_trace()
            num_chars = sum(len(w) for w in words)
            num_chars_total = len(tweet)
            num_terms = len(tweet.split())
            num_words = len(words.split())
            avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
            num_unique_terms = len(set(words.split()))

            ###Modified FK grade, where avg words per sentence is just num words/1
            FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
            ##Modified FRE score, where sentence fixed to 1
            FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

            twitter_objs = self.count_twitter_objs(tweet)
            retweet = 0
            if "rt" in words:
                retweet = 1
            features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                        num_unique_terms, sentiment[2][0].tolist(), sentiment[2][2].tolist(), sentiment[2][1].tolist(), sentiment[2][1].tolist()-sentiment[2][0].tolist()+sentiment[2][1].tolist(),
                        twitter_objs[2], twitter_objs[1],
                        twitter_objs[0], retweet]
            #features = pandas.DataFrame(features)
            return features









    def get_feature_array(self,tweets):
        feats=[]
        print(self.lang)
        for t in tweets:
            feats.append(self.other_features(t))
        # return np.array(feats)
        other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                                "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", \
                                "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

        tfidf,_,vocab = self.get_tfidf(tweets)
        
        pos,pos_vocab = self.get_pos(tweets)

        M = np.concatenate([tfidf,pos,feats],axis=1)

        variables = ['']*len(vocab)
        for k,v in vocab.items():
            variables[v] = k

        pos_variables = ['']*len(pos_vocab)
        for k,v in pos_vocab.items():
            pos_variables[v] = k

        feature_names = variables+pos_variables+other_features_names

        return M,feature_names


if __name__ == '__main__':
    nltk.download('averaged_perceptron_tagger')
    # fe = features(lang='en')

    # tweets = ["!!!!! RT @mleew17: boy dats cold...tyga dwn bad we'll for cuffin dat hoes in the 1st place!! #notrump =/ 😀", 
    #           '!!!!!!! RT @UrKindOfBrand Dawg!!!! RT @80sbaby4life: You ever fuck a bitch and she start to cry? You be confused as shit',
    #           '!!!!!!!!! RT @C_G_Anderson: @viva_based she look like a tranny',
    #           '!!!!!!!!!!!!! RT @ShenikaRoberts: The shit you hear about me might be true or it might be faker than the bitch who told it to ya &#57361;', 
    #           '!!!!!!!!!!!!!!!!!!"@T_Madison_x: The shit just blows me..claim you so faithful and down for somebody but still fucking with hoes! &#128514;&#128514;&#128514;"', 
    #           '!!!!!!"@__BrighterDays: I can not just sit up and HATE on another bitch .. I got too much shit going on!"',
    #           "!!!!&#8220;@selfiequeenbri: cause I'm tired of you big bitches coming for us skinny girls!!&#8221;", '" &amp; you might not get ya bitch back &amp; thats that "',
    #           '" @rhythmixx_ :hobbies include: fighting Mariam"\n\nbitch',
    #           '" Keeks is a bitch she curves everyone " lol I walked into a conversation like this. Smh']
    # (M,names)=fe.get_feature_array(tweets)
    # print(M[6],len(names))
    fe = features(lang='hi')
    tweets = ["वाशिंगटन: दुनिया के सबसे शक्तिशाली देश के राष्ट्रपति बराक ओबामा ने प्रधानमंत्री नरेंद्र मोदी के संदर्भ में 'टाइम' पत्रिका में लिखा, नरेंद्र मोदी ने अपने बाल्यकाल में अपने परिवार की सहायता करने के लिए अपने पिता की चाय बेचने में मदद की थी। आज वह दुनिया के सबसे बड़े लोकतंत्र के नेता हैं और गरीबी से प्रधानमंत्री तक की उनकी जिंदगी की कहानी भारत के उदय की गतिशीलता और क्षमता को परिलक्षित करती है।"]    
    (M,names)=fe.get_feature_array(tweets)
    print(M,len(names))