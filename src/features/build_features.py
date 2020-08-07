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
import io
import pandas as pd
import numpy as np

BUCKET = 'tweets-1301' # s3 bucket name

class TrainingDataMaker():

    '''Loads up the data for training the models, from Social Media for Public Health, downloaded to s3.
    This dataset contains contains annotations for whether a tweet is relevant to the topic of flu vaccination
    and if the author intends to receive a flu vaccine. Analysis of this dataset was published in:

    Xiaolei Huang, Michael C. Smith, Michael Paul, Dmytro Ryzhkov, Sandra Quinn, David Broniatowski, Mark Dredze. Examining Patterns of Influenza Vaccination in Social Media. 
    AAAI Joint Workshop on Health Intelligence (W3PHIAI), 2017.'''
    
    def __init__(self, logger):

        self.logger = logger
        self.s3_client = boto3.client('s3')
        self.raw_data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('raw').resolve()

    def get_data(self):

        '''Reads in gzip JSON from s3, saves JSON locally.'''
        obj = self.s3_client.get_object(Bucket=BUCKET, Key='flu_vaccine.json_.gz')
        obj = io.BytesIO(obj['Body'].read())
        fname = self.raw_data_path.joinpath('data.json').resolve()

        with gzip.open(obj, 'rb') as f_in:
            with open(fname, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                f_out.close()

        self.logger.info('loaded data, now cleaning...')
    
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
            'is_truncated': [],
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
        print(self.raw_df.head())

    def execute_dataprep(self):

        #self.get_data()
        self.clean_data()


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

    main()
