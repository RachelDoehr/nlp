import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from plotly import graph_objs as go
import plotly.express as px
from plotly.graph_objs import *
from datetime import datetime as dt

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
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# -------------------------------- SETUP, SUPPORTING FUNCTIONS ------------------

ALL_KEYS = []
with open('ALL_KEYS.pkl', 'rb') as f:
    ALL_KEYS.extend(pickle.load(f))
    f.close()

# create drop down menu
OPTS_KEYS = []
for word in ALL_KEYS[0]:
    OPTS_KEYS.append(
        {
            'label': word,
            'value': word
        }
    )

def visualize_umap_embeddings(keys, selected_idx):
    
        '''Creates visualization of the learned umap + word2vec clusters'''

        # load full possible vocubulary's embeddings
        mdl = 'covid_'
        output=[]
        with open(mdl+'embeddings_clusters.npy', 'rb') as f:
            output.append(np.load(f))
            f.close()
        with open(mdl+'word_clusters.pkl', 'rb') as f:
            output.append(pickle.load(f))
            f.close()

        # get only the selected words
        filtered_output = []
        filtered_output.append(output[0][selected_idx, :, :])
        filtered_output.append(itemgetter(*selected_idx)(output[1]))

        def umap_plot_similar_words(inputs):

            layout = Layout(
                title='Word Embeddings',
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                width=1000,
                height=800,
                margin=dict(
                    l=50,
                    r=50,
                    b=50,
                    t=50,
                    pad=2
                ),
                xaxis=dict(showgrid=False, zeroline=False, zerolinewidth=1, zerolinecolor='DarkGray'),
                yaxis=dict(showgrid=False, zeroline=False, zerolinewidth=1, zerolinecolor='DarkGray')
            )

            fig = go.Figure(layout=layout)

            colors = ['deepskyblue', 'greenyellow']
            for key, embeddings, words, color in zip(keys, inputs[0], inputs[1], colors):
                x = embeddings[:, 0]
                y = embeddings[:, 1]

                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    name=key,
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=8,
                        line=dict(
                            color=color,
                            width=1
                        )
                    ),
                ))
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y*1.008,
                    name='textlabels',
                    mode='text',
                    text=words
                ))
            fig.update_traces(
                textposition='top center',
                textfont_size=8,
                textfont_color='DarkGray'
            )
            fig.update_layout(
                legend=dict(
                    font=dict(
                        size=10,
                        color="DarkGray"
                    )
                )
            )
            # set showlegend property by name of trace
            for trace in fig['data']: 
                if(trace['name'] == 'textlabels'): trace['showlegend'] = False
            return fig
        
        fig = umap_plot_similar_words(filtered_output)
        return fig

def filter_viz(selected_key1, selected_key2):

    '''Runs the visualization for selected keys'''

    selected_keys = [selected_key1, selected_key2]

    selected_indices = [i for i, j in enumerate(ALL_KEYS[0]) if j in selected_keys]
    fig = visualize_umap_embeddings(keys=selected_keys, selected_idx=selected_indices)
    return fig


# --------------------------- LAYOUT OF APP ---------------------------

# Layout of Dash App
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.H1("VACCINE TWEETS: COVID-19"),
                        html.P(
                            """Select different words to visualize using the input boxes below."""
                        ),
                        # Change to side-by-side for mobile layout
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown to select times
                                        dcc.Dropdown(
                                            id="word-selector-1",
                                            multi=False,
                                            options=OPTS_KEYS,
                                            placeholder="Select first word",
                                            value='fauci'
                                        ),
                                        html.P(' '),
                                        html.P('Select second word to compare: '),
                                        dcc.Dropdown(
                                            id="word-selector-2",
                                            multi=False,
                                            options=OPTS_KEYS,
                                            placeholder="Select next word",
                                            value='microchip'
                                        )
                                    ],
                                ),
                            ],
                        ),
                        html.P(' '),
                        html.P(' '),
                        html.P(' '),
                        html.Div([
                            html.Div("""The scatter plot to the right depicts the learned vector embeddings of 
                                a Doc2Vec distributed memory model on a corpus of ~400,000 Tweets that 
                                mention vaccines, sampled from July - August 2020.""",
                                style={'fontSize': 9}
                            ),
                            html.Br(),
                            html.Div("""
                                The 25 most similar words are found for the selected words of interest,
                                and the vector embeddings run through a dimensionality reduction algorithm
                                (UMAP) to distill them to only 2 dimensions which can be visualized.""",
                                style={'fontSize': 9}
                            ),
                            html.Br(),
                            html.Div("""
                                The distance between words on the graph can be rougly interpreted as how similar
                                the learned representations of them are. Research is ongoing into if UMAP preserves
                                global structure (green dots' distance from blue dots') as well as local (how
                                close the green or blue dots are to each other).""",
                                style={'fontSize': 9}
                            )
                            ],
                        style={'marginBottom': 50, 'marginTop': 25}
                        )
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id="covid-graph")
                    ],
                ),
            ],
        )
    ]
)

# ----------------------------------------- CALLBACKS -----------

@app.callback(
    dash.dependencies.Output('covid-graph', 'figure'),
    [
        dash.dependencies.Input('word-selector-1', 'value'),
        dash.dependencies.Input('word-selector-2', 'value')
    ]
)

def push_covid_graph(value1, value2):
    return filter_viz(value1, value2)


if __name__ == "__main__":
    app.run_server(debug=True)
