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
import ast
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import preprocessor as p
from nltk.corpus import stopwords
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import string
from nltk import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn import utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from gensim.models import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument
import multiprocessing

from sklearn.manifold import TSNE
import umap

cores = multiprocessing.cpu_count()
tqdm.pandas(desc="progress-bar")

BUCKET = 'tweets-1301' # s3 bucket name

class NLPClassifier():

    '''Loads up preprocessed tweet csv data.
    
    Trains 2 doc2vec models, one for the flu vaccine tweets data from 'normal' times.
    Second model is on the current covid vaccine tweets. Stores in s3.
    '''
    
    def __init__(self, logger):

        self.logger = logger
        sns.set(style="white")
        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()
        self.models_path = Path(__file__).resolve().parents[2].joinpath('models').resolve()

        self.vectors = {}
        self.covid_predictions = {}

    def load_data(self):

        '''Reads in csvs from local files'''

        self.features_train_df = pd.read_csv(self.data_path.joinpath('training_nlp.csv'))
        self.features_cv19_df = pd.read_csv(self.data_path.joinpath('cv19_nlp.csv'))
        self.logger.info('loaded data...')

    def fit_doc2vec(self, label, root_form, df_train):

        '''Fits doc2vec word embedding model on vaccine tweets. Requires specification
        of which root form of the words to use'''

        df = df_train[[root_form, 'tweet_id']]
        df['sent'] = df[root_form].apply(lambda x: ast.literal_eval(x))
        df['sent'] = df['sent'].str.join(' ')
        df['tag'] = str(df.tweet_id)

        def prep_text(text):
            tokens = []
            for t in text.split(' '):
                tokens.append(t)
            return tokens

        # tag document
        train_tagged = df.apply(
            lambda x: TaggedDocument(words=prep_text(x['sent']), tags=[x.tag]), axis=1)

        # build vocabulary
        model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=10, negative=5, min_count=1, workers=cores, alpha=0.065, min_alpha=0.065)
        model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

        # train embeddings
        for epoch in range(30):
            model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
            model_dmm.alpha -= 0.002
            model_dmm.min_alpha = model_dmm.alpha

        # infer the word vectors using trained model
        def vec_for_learning(model, tagged_docs):
            sents = tagged_docs.values
            targets, vects = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
            return targets, vects

        # upload gzip'd model to s3
        model_dmm.save(label+'.pkl.gz')
        key = label+'.pkl.gz'

        #self.s3_client.put_object(Bucket=BUCKET, Body='model_dmm_covid.pkl.gz', Key=key)

        self.logger.info('fit word2vec model for covid-19 vaccine tweets, saved to s3...')
    
    def umap_on_vectors(self, mdl, keys):

        '''Loads trained models and word embeddings for covid-19 vaccine tweets and prior to
        doing any visualization.'''

        model = Doc2Vec.load(mdl)

        # get the embeddings and embeddings for the most similar words
        self.embedding_clusters = []
        self.word_clusters = []
        for word in keys:
            embeddings = []
            words = []
            for similar_word, _ in model.most_similar(word, topn=25):
                words.append(similar_word)
                embeddings.append(model[similar_word])
            self.embedding_clusters.append(embeddings)
            self.word_clusters.append(words)
        
        # run umap
        self.embedding_clusters = np.array(self.embedding_clusters)
        n, m, k = self.embedding_clusters.shape

        self.umap_model_en_2d = umap.UMAP(n_neighbors=10, random_state=42)

        self.embeddings_en_2d_umap = np.array(self.umap_model_en_2d.fit_transform(self.embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
        
        self.logger.info('selected interesting word vectors and ran umap...')

    def visualize_umap_embeddings(self, keys, fname):

        '''Creates visualization of the learned umap + word2vec clusters'''

        def umap_plot_similar_words(title, labels, embedding_clusters, word_clusters, filename=None):

            plt.figure(figsize=(16, 9))
            colors = cm.rainbow(np.linspace(0, 1, len(labels)))
            for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
                x = embeddings[:, 0]
                y = embeddings[:, 1]
                plt.scatter(x, y, c=color, alpha=0.7, label=label)
                for i, word in enumerate(words):
                    plt.annotate(word, alpha=0.8, xy=(x[i], y[i]), xytext=(5, 2),
                                textcoords='offset points', ha='right', va='bottom', size=8)
            plt.legend(loc=4)
            plt.title(title)
            plt.grid(True)
            if filename:
                plt.savefig(self.graphics_path.joinpath(filename).resolve(), format='png', dpi=150, bbox_inches='tight')
            plt.show()
        
        umap_plot_similar_words('Similar words from COVID-19 Vaccine Tweets', keys, self.embeddings_en_2d_umap, self.word_clusters, fname)
    
    def execute_models(self):
    
        self.load_data()
        self.fit_doc2vec(
            label='model_dmm_covid.model',
            root_form='tweet_tokens_formal',
            df_train=self.features_cv19_df
        )
        # self.fit_doc2vec(
        #     label='model_dmm_flu.model',
        #     root_form='tweet_tokens_formal',
        #     df_train=self.features_train_df
        # )
        # self.umap_on_vectors(
        #     mdl='model_dmm_covid.model.pkl.gz',
        #     keys=['trump', 'safety', 'russia', 'rush', 'autism']
        # )
        # self.visualize_umap_embeddings(
        #     keys=['trump', 'safety', 'russia', 'rush', 'autism'],
        #     fname='similar_words_covid.png'
        # )
        # self.umap_on_vectors(
        #     mdl='model_dmm_flu.model.pkl.gz',
        #     keys=['trump', 'safety', 'russia', 'rush', 'autism']
        # )
        # self.visualize_umap_embeddings(
        #     keys=['trump', 'safety', 'russia', 'rush', 'autism'],
        #     fname='similar_words_flu.png'
        # )

def main():
    """ Runs model training processes and saves errors in /reports/figures.
    """
    logger = logging.getLogger(__name__)
    logger.info('training the models...')

    Models = NLPClassifier(logger)
    Models.execute_models()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())


    main()
