# -*- coding: utf-8 -*-
from pathlib import Path
from io import StringIO
import logging
from dotenv import find_dotenv, load_dotenv
import boto3
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from cycler import cycler
import datetime
import io
from functools import reduce
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

BUCKET = 'tweets-1301' # s3 bucket name

class RawVisualizer():

    '''Loads up preprocessed tweet csv data.
    Performs initial data exploratory visualizations to understand what we are working with.
    '''
    
    def __init__(self, logger):

        self.logger = logger
        sns.set(style="white")
        register_matplotlib_converters()
        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()

    def load_data(self):

        '''Reads in csvs from local files'''

        self.features_train_df = pd.read_csv(self.data_path.joinpath('training_nlp.csv'))
        self.features_cv19_df = pd.read_csv(self.data_path.joinpath('cv19_nlp.csv'))
        self.logger.info('loaded data...')

   def execute_dataviz(self):
    
        self.load_data()



def main():
    """ Runs data processing scripts to turn raw data from s3 into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('visualizing the datasets...')

    Viz = RawVisualizer(logger)
    Viz.execute_dataviz()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())


    main()
