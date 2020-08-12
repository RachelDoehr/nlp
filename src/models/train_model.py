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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score, plot_confusion_matrix
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from gensim.models import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument
import multiprocessing

from sklearn.manifold import TSNE

cores = multiprocessing.cpu_count()
tqdm.pandas(desc="progress-bar")

BUCKET = 'tweets-1301' # s3 bucket name

class NLPClassifier():

    '''Loads up preprocessed tweet csv data.
    
    [tbd].
    '''
    
    def __init__(self, logger):

        self.logger = logger
        sns.set(style="white")
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
            'fitted_covid_idf_matrix': tfidf_vectorizer.fit_transform(X_covid),
            'transformed_flu_vectors_train': fitted_tfidf_vectorizer.transform(X_train_flu), # for training
            'transformed_flu_vectors_test': fitted_tfidf_vectorizer.transform(X_test_flu), # for tuning
            'transformed_vectors_covid': fitted_tfidf_vectorizer.transform(X_covid)
        }
        self.logger.info('fit tf-idf model and stored vectors...')
    
    def explore_cv19_tfidf(self, root_form):

        '''Visualize the top terms in the doc fitted in tf-idf'''

        vect = TfidfVectorizer(use_idf=True, ngram_range=(1, 1), max_features=200, stop_words='english')
        X_covid =  self.features_cv19_df[root_form].str.join('')

        tfidf_matrix = vect.fit_transform(X_covid)
        df = pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names())
        
        top_feats = pd.DataFrame(df.sum(numeric_only=True).sort_values()).tail(30)
        top_feats.columns = ['idf_score']
        top_feats.plot(kind='bar')
        plt.show()

    def fit_doc2vec_cv19(self, label, root_form):

        '''Fits doc2vec word embedding model on covid-19 vaccine tweets. Requires specification
        of which root form of the words to use'''

        df = self.features_cv19_df[[root_form, 'tweet_id']]
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

        model_dmm.save('model_dmm.model')

        #lbl_train_dmm, X_train_dmm = vec_for_learning(model_dmm, train_tagged)

        self.logger.info('fit word2vec model for covid-19 vaccine tweets, saved locally...')
    
    def tsne_on_vectors(self, mdl, keys):

        '''Loads trained models and word embeddings for covid-19 vaccine tweets and prior to
        doing any visualization.'''

        model = Doc2Vec.load(mdl)

        # get the embeddings and embeddings for the most similar words
        self.embedding_clusters = []
        self.word_clusters = []
        for word in keys:
            embeddings = []
            words = []
            for similar_word, _ in model.most_similar(word, topn=30):
                words.append(similar_word)
                embeddings.append(model[similar_word])
            self.embedding_clusters.append(embeddings)
            self.word_clusters.append(words)
        
        # run t-sne
        self.embedding_clusters = np.array(self.embedding_clusters)
        n, m, k = self.embedding_clusters.shape
        tsne_model_en_2d = TSNE(perplexity=55, n_components=2, init='pca', n_iter=3500, random_state=42)
        self.embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(self.embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
        
        self.logger.info('selected interesting word vectors and ran t-sne...')

    def visualize_tsne_embeddings(self, keys):

        '''Creates visualization of the learned tnse + word2vec clusters'''

        def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, filename=None):

            plt.figure(figsize=(16, 9))
            colors = cm.rainbow(np.linspace(0, 1, len(labels)))
            for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
                x = embeddings[:, 0]
                y = embeddings[:, 1]
                plt.scatter(x, y, c=color, alpha=0.7, label=label)
                for i, word in enumerate(words):
                    plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                                textcoords='offset points', ha='right', va='bottom', size=8)
            plt.legend(loc=4)
            plt.title(title)
            plt.grid(True)
            if filename:
                plt.savefig(self.graphics_path.joinpath(filename).resolve(), format='png', dpi=150, bbox_inches='tight')
            plt.show()
        
        tsne_plot_similar_words('Similar words from COVID-19 Vaccine Tweets', keys, self.embeddings_en_2d, self.word_clusters, 'similar_words.png')


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
            label='flu_intention__tfidf_lemmas',
            root_form='tweet_tokens_formal',
            flu_df=self.flu_df_intention
        )
        # self.fit_SGD(
        #     data_label='flu_intention__tfidf_tokens',
        #     conf_mat_lbl='flu_intention_tfidf_tokens',
        #     class_labels=['No', 'Yes, Intend']
        # )
        #self.explore_cv19_tfidf(root_form='tweet_lemmas_formal')
        # self.fit_doc2vec_cv19(
        #     root_form='tweet_tokens_formal',
        #     label='covid_vaccines__doc2vec_tokens'
        # )
        self.fit_doc2vec_cv19(
            label='dmm_covid.model',
            root_form='tweet_tokens_formal'
        )
        self.tsne_on_vectors(
            mdl='model_dmm.model',
            keys=['trump', 'fauci', 'safety', 'microchip', 'volunteer', 'rush', 'biden']
        )
        self.visualize_tsne_embeddings(
            keys=['trump', 'fauci', 'safety', 'microchip', 'volunteer', 'rush', 'biden']
        )

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
