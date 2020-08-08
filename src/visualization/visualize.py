# -*- coding: utf-8 -*-
from pathlib import Path
from io import StringIO
import logging
from dotenv import find_dotenv, load_dotenv
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
import plotly.graph_objects as go
import plotly.io as pio

print(pio.renderers.default)

BUCKET = 'tweets-1301' # s3 bucket name

class RawVisualizer():

    '''Loads up preprocessed tweet csv data.
    Performs initial data exploratory visualizations to understand what we are working with.
    '''
    
    def __init__(self, logger):

        self.logger = logger
        sns.set(style="white")
        register_matplotlib_converters()
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()

    def load_data(self):

        '''Reads in csvs from local files'''

        self.features_train_df = pd.read_csv(self.data_path.joinpath('training_nlp.csv'))
        self.features_cv19_df = pd.read_csv(self.data_path.joinpath('cv19_nlp.csv'))
        self.logger.info('loaded data...')
    
    def class_distributions(self):

        '''Training data, what the breakdown of the target(s) are'''

        # Create dimensions
        class_dim = go.parcats.Dimension(values=self.features_train_df.FLU_vac_relevant, label="Relevant")

        gender_dim = go.parcats.Dimension(values=self.features_train_df.FLU_vac_sentiment, label="Sentiment")

        survival_dim = go.parcats.Dimension(
            values=self.features_train_df.FLU_vac_intend_to_receive, label="Intend to Receive Flu Vaccine"
        )
        received_dim = go.parcats.Dimension(
            values=self.features_train_df.FLU_vac_received, label="Received Flu Vaccine"
        )

        # Create parcats trace
        self.features_train_df['Color'] = 1
        self.features_train_df.loc[self.features_train_df.FLU_vac_received == 'received', 'Color'] = 0

        color = self.features_train_df.Color
        colorscale = [[0, 'crimson'], [1, 'deepskyblue']]

        fig = go.Figure(data = [go.Parcats(dimensions=[class_dim, gender_dim, survival_dim, received_dim],
                line={'color': color, 'colorscale': colorscale},
                hoveron='color', hoverinfo='count+probability',
                labelfont={'size': 18},
                tickfont={'size': 16},
                arrangement='freeform')])

        fig.write_html('tmp.html', auto_open=True)

    def execute_dataviz(self):
    
        self.load_data()
        self.class_distributions()

def main():
    """ Runs data visualization processes and saves in /reports/figures.
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
