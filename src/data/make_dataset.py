# -*- coding: utf-8 -*-

'''This program is for streaming Twitter data to a PostgresSQL DB on AWS. The tweets are currently filtered to COVID-19 related.'''

import json
import os
import csv
from datetime import datetime
import sys
from tweepy.streaming import StreamListener
import tweepy
import pandas as pd
import boto3
import numpy as np
import argparse
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import time
import sqlalchemy
from sqlalchemy import create_engine

while True:

    class Listener(StreamListener):

        def __init__(self, output_file=sys.stdout, time_limit=60):
            self.start_time = time.time()
            self.output_file = output_file
            self.limit = time_limit
            self.output_json = []
            super(Listener,self).__init__()

        def on_data(self, data):
            if (time.time() - self.start_time) < self.limit:
                if hasattr(data, 'retweeted_status'):
                    print('retweet')
                else:
                    try:
                        all_data = json.loads(data)
                        self.output_json.append(all_data)
                        return True
                    except KeyError:
                        print('Key error')
                        return True
                
            else:
                # process and upload
                
                def process_tweet(all_data):
                    tweet_DICT = {}
                    out_dic = {}
                    try:
                        tweet_DICT['tweet_text'] = all_data["full_text"]
                    except KeyError:
                        try:
                            tweet_DICT['tweet_text'] = all_data["extended_tweet"]["full_text"]
                        except KeyError:
                            try:
                                tweet_DICT['tweet_text'] = all_data["text"]
                            except:
                                pass
                    try:
                        tweet_DICT['tweet_id'] = all_data["id_str"]
                        tweet_DICT['created_at'] = datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S')
                        user = all_data.get("user")
                                
                        tweet_DICT['place_full_name'] = None
                        tweet_DICT['place_country_code'] = None

                        if user is not None:
                            tweet_DICT['location_user'] = all_data.get("user").get("location")

                        place = all_data.get("place")
                        if place is not None:
                            tweet_DICT['place_full_name'] = all_data.get("place").get("full_name") 
                            tweet_DICT['place_country_code'] = all_data.get("place").get("country_code")
                    except:
                        print()
                    try:
                        if 'RT @' not in tweet_DICT['tweet_text']:
                            out_dic = tweet_DICT
                        else:
                            out_dic = {}
                    except KeyError:
                        pass
                    return out_dic
                        
                tweet_list = [process_tweet(jj) for jj in self.output_json if jj != {}]

                df = pd.DataFrame(tweet_list)
                df.dropna(subset=['tweet_text'], inplace=True)
                print("Collected ", df.shape, " tweets")
                db_string = "postgresql+psycopg2://postgres:" + POSTGRES_PASSWORD + "@database.cgiiehbwlxvq.us-east-2.rds.amazonaws.com"
                engine = create_engine(db_string)
                df.to_sql(TBL, engine, schema='twitter', if_exists='append', index=False)
                print('uploaded')
                return False

        def on_error(self, status_code):
            if (time.time() - self.start_time) < self.limit:
                if status_code == 420:
                    return False
                else:
                    print(status_code)
                    return True
            else:
                return False

    class TweetStreamer():

        def __init__(self):

            self.mybucketname = 'twitterdata001'
            self.s3client = boto3.client('s3')

            self.auth = tweepy.OAuthHandler(CONSUMER_API_KEY, CONSUMER_API_SECRET_KEY)
            self.auth.set_access_token(ACCESS_KEY, ACCESS_SECRET_TOKEN)
            self.api = tweepy.API(self.auth, 
                                wait_on_rate_limit=True,
                                wait_on_rate_limit_notify=True)

        def start_streaming(self):

            listener = Listener(time_limit=60*5)

            stream = tweepy.Stream(auth=self.api.auth, listener=listener, tweet_mode='extended')
            stream.filter(languages=['en'], track=ALL_WORDS) 
            try:
                print('Started streaming.')
                stream.sample(languages=['en'])
            except KeyboardInterrupt:
                print("Stopped.")
            finally:
                print('Done.')
                stream.disconnect()
            
        def exec_stream(self):

            self.start_streaming()
        
    def main():

        DataStreamer = TweetStreamer()
        DataStreamer.exec_stream()

    if __name__ == '__main__':

        # not used in this stub but often useful for finding various files
        project_dir = Path(__file__).resolve().parents[3]

        # find .env automagically by walking up directories until it's found, then load up the .env entries as environment variables
        load_dotenv(find_dotenv())

        ACCESS_KEY = os.getenv("ACCESS_KEY")
        ACCESS_SECRET_TOKEN = os.getenv("ACCESS_SECRET_TOKEN")
        CONSUMER_API_KEY = os.getenv("CONSUMER_API_KEY")
        CONSUMER_API_SECRET_KEY = os.getenv("CONSUMER_API_SECRET_KEY")
        POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

        parser = argparse.ArgumentParser()
        parser.add_argument("table", action="store", help="PostgresSQL table to write to. Either: 'streamed_full' or 'streamed_vaccine'")
        parser.add_argument("tweet_words", action="store", help="Which words to include in tweets. Currently either: 'covid' or 'vaccines'")
        results = parser.parse_args()

        TBL = results.table
        WRDS = results.tweet_words
        if WRDS == 'covid':
            ALL_WORDS = ["covid", "coronavirus", "COVID-19", "Covid", "corona virus", "Corona virus", "Coronavirus"]
        elif WRDS == 'vaccines':
            ALL_WORDS = ['vaccine', 'vaccines', 'immunization', 'immunity']
        else:
            sys.exit()
        
        main()
