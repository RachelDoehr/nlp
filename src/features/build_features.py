# -*- coding: utf-8 -*-
from pathlib import Path
from io import StringIO
import logging
from dotenv import find_dotenv, load_dotenv
import boto3
import json
import datetime
import gzip
import shutil
import re
import os
import io
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import preprocessor as p
from nltk.corpus import stopwords
import spacy
import nltk
from nltk.tokenize import TweetTokenizer
import string
from nltk import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

BUCKET = 'tweets-1301' # s3 bucket name

class TrainingDataMaker():

    '''Loads up the data for training the models, from Social Media for Public Health, downloaded to s3.
    This dataset contains contains annotations for whether a tweet is relevant to the topic of flu vaccination
    and if the author intends to receive a flu vaccine. Analysis of this dataset was published in:

    Xiaolei Huang, Michael C. Smith, Michael Paul, Dmytro Ryzhkov, Sandra Quinn, David Broniatowski, Mark Dredze. Examining Patterns of Influenza Vaccination in Social Media. 
    AAAI Joint Workshop on Health Intelligence (W3PHIAI), 2017.
    
    Performs preprocessing.'''
    
    def __init__(self, logger):

        self.logger = logger
        self.s3_client = boto3.client('s3')
        self.raw_data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('raw').resolve()
        self.interim_data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('interim').resolve()
        self.processed_data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()

        oov_df = pd.read_csv(self.raw_data_path.joinpath('OOV_dict.csv')) # load the out of vocabulary file
        self.oov_informal = oov_df.informal.to_list()
        self.oov_formal = oov_df.translated.to_list()

        self.nlp = spacy.load("en_core_web_sm")
        self.ps = PorterStemmer()

        # tweet preprocessing setup ----------------------------------------
        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY) # get rid of everything but hashtags and numbers

    def get_training_data(self):

        '''Reads in gzip JSON from s3, saves JSON locally.'''
        obj = self.s3_client.get_object(Bucket=BUCKET, Key='flu_vaccine.json_.gz')
        obj = io.BytesIO(obj['Body'].read())
        fname = self.raw_data_path.joinpath('data.json').resolve()

        with gzip.open(obj, 'rb') as f_in:
            with open(fname, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                f_out.close()

        self.logger.info('loaded training data, now cleaning...')

    def get_covid19_vac_tweets(self):

        '''Queries the PostresSQL DB on AWS to get the COVID-19 incoming tweets.
        The DB is constantly streaming them in so for testing purposes here have put a limit of 5,000 for speed.
        
        [Considering productionizing this part later if I make a dashboard]'''

        db_string = "postgresql+psycopg2://postgres:" + POSTGRES_PASSWORD + "@database.cgiiehbwlxvq.us-east-2.rds.amazonaws.com"
        engine = create_engine(db_string)

        self.raw_cv19_df = pd.read_sql(
            "SELECT * FROM twitter.streamed_vaccine LIMIT 1000",
            con=engine
        )

    def clean_data(self):

        '''Turns JSON into prepped dataframe with needed info'''

        tweets = []
        for line in open(self.raw_data_path.joinpath('data.json').resolve(), 'r', encoding='utf8'):
            tweets.append(json.loads(line))

        data = [json.loads(line) for line in open(self.raw_data_path.joinpath('data.json').resolve(), 'r', encoding='utf8')]
        D = {
            'tweet_id': [],
            'tweet_text': [],
            'is_retweet': [],
            'FLU_vac_intend_to_receive': [],
            'FLU_vac_received': [],
            'FLU_vac_relevant': [],
            'FLU_vac_sentiment': []
        }
        
        for idx in range(0, len(data)):

            d = json.dumps(data[idx])
            loaded_d = json.loads(d)

            D['tweet_id'].append(loaded_d['id'])
            D['tweet_text'].append(loaded_d['tweet']['text'])
            D['is_retweet'].append(loaded_d['tweet']['retweeted'])
            try:
                D['FLU_vac_intend_to_receive'].append(loaded_d['label']["flu_vaccine_intent_to_receive"])
            except KeyError:
                D['FLU_vac_intend_to_receive'].append('NA')
            try:
                D['FLU_vac_received'].append(loaded_d['label']["flu_vaccine_received"])
            except KeyError:
                D['FLU_vac_received'].append('NA')
            try:
                D['FLU_vac_relevant'].append(loaded_d['label']["flu_vaccine_relevant"])
            except KeyError:
                D['FLU_vac_relevant'].append('NA')
            try:
                D['FLU_vac_sentiment'].append(loaded_d['label']["flu_vaccine_sentiment"])
            except KeyError:
                D['FLU_vac_sentiment'].append('NA')
        
        self.raw_df = pd.DataFrame.from_dict(D, orient='index').transpose()
        self.raw_df.to_csv(self.interim_data_path.joinpath('raw_df.csv').resolve())
        self.logger.info('handled JSON -> CSV conversion, saved locally...')

    def preprocess(self, t, lowercase=True):

        '''Initial removal of strange Tweet characters'''

        processed_tweet = p.clean(t)

        # Remove all the special characters
        processed_tweet = re.sub(r'\W', ' ', processed_tweet)

        # remove all single characters
        processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    
        # Remove single characters from the start
        processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
    
        # Substituting multiple spaces with single space
        processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    
        # Removing prefixed 'b'
        processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

        if lowercase:
            processed_tweet = processed_tweet.lower()
        return processed_tweet

    def token_process(self, raw_text):

        """
        Takes in a string of text, then performs the following:
        remove all punctuation, remove all stopwords, returns a list of the cleaned text
        """

        # Check characters to see if they are in punctuation
        nopunc = [char for char in list(raw_text) if char not in string.punctuation]
        # Join the characters again to form the string
        nopunc = ''.join(nopunc)
        # Now just remove any stopwords
        return [word for word in nopunc.lower().split() if word.lower() not in stopwords.words('english')]
    
    def replace(self, token):

        '''Helper function for removing out of vocabulary words'''

        replacement = None
        if token in self.oov_informal:
            replacement = self.oov_formal[self.oov_informal.index(token)]
        else:
            replacement = token
        return replacement
    
    def replace_OOV_words(self, text_token_list):

        '''
        Replace the informal tweet words like B2B, etc. with the human-annotated ones that are most likely to be correct.
        Documentation and OOV dictionary available in https://crisisnlp.qcri.org/lrec2016/lrec2016.html
        '''
    
        text_token_list_formal = [self.replace(t) for t in text_token_list]
        return text_token_list_formal
    
    def _gen_lemmas(self, tweet_col):
        # Parse the sentence using the loaded 'en' model object `nlp` using spacy
        sentence = ' '.join([c for c in tweet_col])
        doc = self.nlp(sentence)
        # Extract the lemma for each token and join
        return [token.lemma_ for token in doc]
    
    def _gen_stems(self, tweet_col):
        return [self.ps.stem(w) for w in tweet_col]

    # ------------------------------------- end Tweet processing setup

    def preprocess_tweet_df(self, df, tweet_text_col, lemma=False):
        
        '''Applies the tweet preprocessing functions in order'''

        out_df = df.copy()

        self.logger.info('preprocessing: removing emojis, strange characters...')
        out_df[tweet_text_col] = out_df[tweet_text_col].apply(str)
        out_df['tweet_clean'] = out_df.apply(lambda L: self.preprocess(L[tweet_text_col]), axis=1)
        self.logger.info('preprocessing: turning into tokens...')
        out_df['tweet_tokens'] = out_df.apply(lambda L: self.token_process(L.tweet_clean), axis=1)
        self.logger.info('preprocessing: translating out-of-vocabulary words...')
        out_df['tweet_tokens_formal'] = out_df.apply(lambda L: self.replace_OOV_words(L.tweet_tokens), axis=1)
        self.logger.info('preprocessing: generating stems...')
        out_df['tweet_stems_formal'] = out_df.apply(lambda L: self._gen_stems(L.tweet_tokens), axis=1)

        if lemma:
            self.logger.info('preprocessing: generating lemmas (this takes time)...')
            out_df['tweet_lemmas_formal'] = out_df.apply(lambda L: self._gen_lemmas(L.tweet_tokens_formal), axis=1)

        return out_df

    def execute_dataprep(self):

        #self.get_training_data()
        self.get_covid19_vac_tweets()
        self.clean_data()
        #self.training_clean_df = self.preprocess_tweet_df(self.raw_df, 'tweet_text', lemma=True)
        #self.training_clean_df.to_csv(self.processed_data_path.joinpath('training_nlp.csv'))
        self.cv19_clean_df = self.preprocess_tweet_df(self.raw_cv19_df, 'tweet_text', lemma=True)
        self.cv19_clean_df.to_csv(self.processed_data_path.joinpath('cv19_nlp.csv'))


def main():
    """ Runs data processing scripts to turn raw data from s3 into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    TrainingData = TrainingDataMaker(logger)
    TrainingData.execute_dataprep()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

    main()
