'''This is a supporting file that is to be run before initializing the Dash app. The
doc2vec models are too heavy to hold in memory while running a Dash app, so this loads in the model,
finds all possible keys (words) within a top threshold a user may want to graph, and runs the embeddings through UMAP and saves.'''

import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO, BytesIO
import logging
from dotenv import find_dotenv, load_dotenv
import boto3
import json
import datetime
import gzip
import shutil
from operator import itemgetter 
import re
import pickle
import os
import io
import ast
import pandas as pd
import numpy as np

import nltk
import string
from nltk import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from gensim.models import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
import umap

BUCKET = 'tweets-1301' # s3 bucket name

def load_doc2vec_model(key, key_trainables='', key_vectors=''):

    '''Loads pre-trained doc2vec embedding from s3'''

    session = boto3.Session()
    s3 = session.resource('s3')

    s3.meta.client.download_file(BUCKET, key, str(key))
    if key_trainables != '':
        s3.meta.client.download_file(BUCKET, key_trainables, str(key_trainables))
        s3.meta.client.download_file(BUCKET, key_vectors, str(key_vectors))

    return Doc2Vec.load(str(key))
    
MODEL_COVID = load_doc2vec_model(
    key='model_dmm_covid.model.pkl.gz',
    key_trainables='model_dmm_covid.model.pkl.gz.trainables.syn1neg.npz',
    key_vectors='model_dmm_covid.model.pkl.gz.wv.vectors.npz'
)
MODEL_FLU = load_doc2vec_model(
    key='model_dmm_flu.model.pkl.gz',
)

# most frequent keys to par down
ALL_KEYS = [['trump', 'fauci', 'microchip', 'safety', 'biden', 'russia', 'hoax', 'hcq'],
['trump', 'safety', 'russia', 'rush', 'autism', 'testing', 'health', 'kids', 'doctor']]
with open('ALL_KEYS.pkl', 'wb') as f:
    pickle.dump(ALL_KEYS, f)
    f.close()

def umap_on_vectors(both_keys):
    
        '''Gets most similar words for the models. Runs UMAP to reduce dimensions to 2.'''

        covid_outputs = {
            "embedding_clusters": [],
            "word_clusters": [],
            "umap_model_en_2d": None,
            "embeddings_en_2d_umap": None
        }
        flu_outputs = {
            "embedding_clusters": [],
            "word_clusters": [],
            "umap_model_en_2d": None,
            "embeddings_en_2d_umap": None
        }

        def _run_each(model, outputs, pfix, keys):
    
            # get the embeddings and embeddings for the most similar words
            for word in keys:
                try:
                    embeddings = []
                    words = []
                    for similar_word, _ in model.most_similar(word, topn=25):
                        words.append(similar_word)
                        embeddings.append(model[similar_word])
                    outputs['embedding_clusters'].append(embeddings)
                    outputs['word_clusters'].append(words)
                except:
                    print(word)
                    print('^^ not found')
            
            # run umap
            outputs['embedding_clusters'] = np.array(outputs['embedding_clusters'])
            n, m, k = outputs['embedding_clusters'].shape

            outputs['umap_model_en_2d'] = umap.UMAP(n_neighbors=10, random_state=42)
            outputs['embeddings_en_2d_umap'] = np.array(outputs['umap_model_en_2d'].fit_transform(outputs['embedding_clusters'].reshape(n * m, k))).reshape(n, m, 2)
            
            with open(pfix+'embeddings_clusters.npy', 'wb') as f:
                np.save(f, outputs['embeddings_en_2d_umap'])
                f.close()

            with open(pfix+'word_clusters.pkl', 'wb') as f:
                pickle.dump(outputs['word_clusters'], f)
                f.close()

        _run_each(MODEL_COVID, covid_outputs, 'covid_', keys=both_keys[0])
        _run_each(MODEL_FLU, flu_outputs, 'flu_', keys=both_keys[1])

umap_on_vectors(both_keys=ALL_KEYS)
