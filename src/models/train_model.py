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

import seaborn as sns
import matplotlib.pyplot as plt

import preprocessor as p
from nltk.corpus import stopwords
import spacy
import nltk
from nltk.tokenize import TweetTokenizer
import string
from nltk import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn import utils
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score, plot_confusion_matrix
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

BUCKET = 'tweets-1301' # s3 bucket name

class NLPClassifier():

    '''Loads up preprocessed tweet csv data.
    Trains several models to predict/classify tweets' intention or lack therof to get a covid-19 vaccine, plus
    their general sentiment on it. Uses a tagged dataset (see build_features.py for link) which is regarding
    the flu (seasonal influenza) vaccine.
    
    Approaches include first encoding the language using either (A) TF-IDF or (B) Doc2Vec, limiting/fitting the vocab of
    the classifier to the COVID-19 tweets given the naturally different language/new terms. Then, training a SVM on the
    vectorized features and comparing performance on classifying the Flu tweets (less important) and how it performs
    with the new COVID-19 tweets (the goal, but difficult to define error metrics).
    '''
    
    def __init__(self, logger):

        self.logger = logger
        sns.set(style="white")
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()

        self.vectors = {}

        self.covid_predictions = {}

    def load_data(self):

        '''Reads in csvs from local files'''

        self.features_train_df = pd.read_csv(self.data_path.joinpath('training_nlp.csv'))
        self.features_cv19_df = pd.read_csv(self.data_path.joinpath('cv19_nlp.csv'))
        self.logger.info('loaded data...')
    
    def isolate_datasets(self):

        '''Creates 2 clean copies of the flu data given that some of the tweets labels are NA'''

        self.flu_df_intention = self.features_train_df[self.features_train_df.FLU_vac_intend_to_receive.isin(['yes', 'no'])]
        self.flu_df_intention['target'] = self.flu_df_intention['FLU_vac_intend_to_receive']

        self.flu_df_sentiment = self.features_train_df[self.features_train_df.FLU_vac_sentiment.isin(['positive', 'negative'])]
        self.flu_df_sentiment['target'] = self.flu_df_sentiment['FLU_vac_sentiment']

        self.logger.info('created cleaned copies of flu data for training...')
    
    def fit_tfidf_flu_cv19(self, label, root_form, flu_df):

        '''Fits tf-idf vectorizer on the covid-19 data and transforms the flu data. Requires specification
        of which flu target (sentiment or intention to receive) to use, and lemma/stem/token form of X
        
        flu_df: [self.flu_df_intention OR self.flu_df_sentiment]
        root_form: ['tweet_tokens_formal' OR 'tweet_stems_formal' OR 'tweet_lemmas_formal']'''
        
        # flu data
        X_flu = flu_df[root_form].str.join('')
        y_flu = flu_df.target
        
        X_train_flu, X_test_flu, y_train_flu, y_test_flu = train_test_split(X_flu, y_flu, test_size=0.3, random_state=42, shuffle=True)

        tfidf_vectorizer=TfidfVectorizer(use_idf=True, ngram_range=(1, 1), max_features=100)
        X_covid =  self.features_cv19_df[root_form].str.join('')

        fitted_tfidf_vectorizer = tfidf_vectorizer.fit(X_train_flu) #X_covid

        self.vectors[label] = {
            'root_form': root_form,
            'train_test_flu': [X_train_flu, X_test_flu, y_train_flu, y_test_flu],
            'X_covid': X_covid,
            'fitted_tfidf_vectorizor': fitted_tfidf_vectorizer,
            'transformed_flu_vectors_train': fitted_tfidf_vectorizer.transform(X_train_flu), # for training
            'transformed_flu_vectors_test': fitted_tfidf_vectorizer.transform(X_test_flu), # for tuning
            'transformed_vectors_covid': fitted_tfidf_vectorizer.transform(X_covid)
        }
        self.logger.info('fit tf-idf model and stored vectors...')
    
    def plot_confusion_matrix(self, y_true, y_pred, classes, name, normalize=False, title=None, cmap='bwr'):
    
        """
        Plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
        """

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(dpi=80)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # Decorations
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        pth = Path(self.graphics_path, 'conf_matrix_'+name).with_suffix('.png')
        plt.savefig(pth)
        plt.close()

    def fit_SGD(self, data_label, conf_mat_lbl, class_labels):

        '''Trains  SGD classifier on flu vaccine data, uses grid search CV on train/test data.
        Reports error metrics on the flu target, though less relevant. Generates predictions for covid-19 vac tweets'''

        transformed_flu_vectors_train = self.vectors[data_label]['transformed_flu_vectors_train']
        y_train_flu = self.vectors[data_label]['train_test_flu'][2]
        transformed_flu_vectors_test = self.vectors[data_label]['transformed_flu_vectors_test']
        y_test_flu = self.vectors[data_label]['train_test_flu'][3]

        transformed_vectors_covid = self.vectors[data_label]['transformed_vectors_covid']

        clf = SGDClassifier(random_state=42)

        # use a full grid over all parameters-hinge less for all
        param_grid = {"max_iter": [1, 5, 10],
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    "penalty": ["none", "l1", "l2"]}

        grid = GridSearchCV(clf, param_grid, refit=True, verbose=3) 

        # fitting the model for grid search 
        grid.fit(transformed_flu_vectors_train, y_train_flu) 

        # print best parameter after tuning 
        print(grid.best_params_) 

        grid_predictions = grid.predict(transformed_flu_vectors_test) 

        # errors for flu 
        print(classification_report(y_test_flu, grid_predictions, target_names=class_labels)) 
        print()
        print('Accuracy score: ', accuracy_score(y_test_flu, grid_predictions, normalize=True))

        kappa = cohen_kappa_score(grid_predictions, y_test_flu, weights='quadratic')
        print('Kappa score: ', kappa)
        self.plot_confusion_matrix(
                y_true=y_test_flu,
                y_pred=grid_predictions,
                classes=class_labels,
                normalize=False,
                name=conf_mat_lbl,
                title='Confusion Matrix: Test Set (flu)'
            )

        self.covid_predictions[data_label+'_SGD'] = pd.DataFrame(grid.predict(transformed_vectors_covid))
        self.logger.info('trained SGD classifier on flu dataset, errors in /reports/figures...')
        
    def execute_models(self):
    
        self.load_data()
        self.isolate_datasets()
        self.fit_tfidf_flu_cv19(
            label='flu_intention__tfidf_tokens',
            root_form='tweet_tokens_formal',
            flu_df=self.flu_df_intention
        )
        self.fit_SGD(
            data_label='flu_intention__tfidf_tokens',
            conf_mat_lbl='flu_intention_tfidf_tokens',
            class_labels=['No', 'Yes, Intend']
        )
        self.covid_predictions['flu_intention__tfidf_tokens_SGD'].to_csv('a.csv')

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
