
![Alt Text](https://github.com/RachelDoehr/nlp/blob/master/reports/figures/example_encoding.gif?raw=true)

# Visualization of COVID-19 Vaccine Tweets Using Doc2Vec

 *A demonstration of NLP techniques and a live Dash application*

**IMPLEMENTATION OF WORD EMBEDDING ON TWITTER DATASET, DIMENSIONALITY REDUCTION ON LEARNED
REPRESENTATION USING UMAP. DASH-PLOTLY APPLICATION FOR VISUALIZATION**

> -> Uses <a href="https://developer.twitter.com/en/docs" target="_blank">Twitter API</a> to stream and build a Tweet dataset of ~0.5 million tweets about vaccines, hosted on AWS

> -> Tweet preprocessing and cleaning done using a combination of Spacy, NLTK, and a module for handling the misc. URLs, emojis, etc. in Twitter data, https://pypi.org/project/tweet-preprocessor/

> -> Post-hoc analysis is done on visualizing the learned representations of select interesting words in the dataset using UMAP to reduce dimensions and Dash/Plotly for creating a live app

![Alt Text](https://github.com/RachelDoehr/nlp/blob/master/reports/figures/dash_shot1.PNG?raw=true)

**Motivation & Project Summary**

This project is a demonstration of using Doc2Vec to model a corpus of text, in this case Twitter tweet data. Several methods exist to encode text data, from the simplest count encoders such as a 'bag of words' counting-based approach to more complicated pipelines of neural networks. Doc2Vec is heavily based on Word2Vec, and treats each tweet as its own document. The Distributed Memory model remembers what is missing from the current context, in this case the 'topic' of a tweet, in addition to representing words as word vectors.

After collecting, storing, and cleaning the tweets, the model is trained and saved. Next, a sample of interesting words are predicted (the feature vectors), in addition to the 25 most similar words. Those 200-width representations are then distilled to 2 dimensions using UMAP, and visualized on a Dash Application. **The final product is available [HERE]. The representations largely make intuitive sense, although more data may improve the model, and given Twitter, a fair amount of noise words still make their way in.**

This project started as a demonstration of using a publicly available dataset of seasonal flu vaccine tweets tagged to whether or not the person planned to get a vaccine to predict how likely people were to get the COVID-19 vaccine. However, it ended up useless not due to ML issues but because **manual reading of a random sample of 1,000 of today's tweets revealed 1 pro-vaccine, "I want to take it as soon as I can" tweet out of 1,000.**

> ***ReadMe Table of Contents***

- INSTALLATION & SETUP
- STRATEGY
- RESULTS
    - EXAMPLES (APP)

---

## Installation & Setup

### Clone

- Clone this repo to your local machine using `https://github.com/RachelDoehr/nlp.git`

### Setup

- Install the required packages

> Requires python3. Suggest the use of an Anaconda environment for easy package management.

```shell
$ pip install -r requirements.txt
```

### Example Run of Processes

- Note that scripts will not be able to run without the password to the Postgres Tweet DB or a Twitter API key, which are stored locally in environment variables here
- Runtimes are significant depending on desired number of tweets to download from the Postgres DB, largely due to lemmatization time (plus to a lesser extent Doc2Vec training time)
- Recommend running the data collection/streaming script as a Cron job that samples Twitter daily and stores to build up sufficient data

```shell
$ python /src/data/make_dataset.py "streamed_vaccine" "vaccines"
$ python /src/features/build_features.py
$ python /src/models/train_model.py
```

To run the Dash App:
```shell
$ nohup python /reports/apps/app.py
```

---

## Strategy

**Preliminary Data Collection**

The data collection process has been running continously for several weeks, yielding ~0.5 million tweets as of late August 2020:

![Alt Text](https://github.com/RachelDoehr/nlp/blob/master/reports/figures/postgresdb.PNG?raw=true)

The tweets had to contain the word 'vaccine' in them to be collected.

**Preprocessing**

Data cleaning included the following:

1) Remove misc. Twitter language (emoji's, hashtags, etc.)
2) Tokenize, remove stop words, make lowercase
3) Lemmatize remaining words

Once preprocessed, we are left with workable tweets and can either proceed with the cleaned tokens or use the lemmas:

| tweet_text                                                                                                                                                                   	| tweet_clean                                                                                                                         	| tweet_tokens                                                                                                               	| tweet_tokens_formal                                                                                                        	| tweet_stems_formal                                                                                                      	| tweet_lemmas_formal                                                                                                                 	|
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|-------------------------------------------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------------------------------------------------------------	|-------------------------------------------------------------------------------------------------------------------------	|-------------------------------------------------------------------------------------------------------------------------------------	|
| Also you may want to   read this.... vaccine manufacturers ARE exempt from liability like I   said....https://t.co/TXZjzsAIE8                                                	| also you may want to   read this vaccine manufacturers are exempt from liability like said                                          	| ['also', 'may', 'want', 'read', 'vaccine',   'manufacturers', 'exempt', 'liability', 'like', 'said']                       	| ['also', 'may', 'want', 'read', 'vaccine',   'manufacturers', 'exempt', 'liability', 'like', 'said']                       	| ['also', 'may', 'want', 'read', 'vaccin',   'manufactur', 'exempt', 'liabil', 'like', 'said']                           	| ['also', 'may', 'want', 'read', 'vaccine',   'manufacturer', 'exempt', 'liability', 'like', 'say']                                  	|
| Russian COVID-19 vaccine \| Sputnik V is   safe and effective, says RDIF CEO: https://t.co/AOlFVh49sR                                                                        	| russian covid 19 vaccine sputnik is safe   and effective says rdif ceo                                                              	| ['russian', 'covid', '19', 'vaccine', 'sputnik', 'safe', 'effective',   'says', 'rdif', 'ceo']                             	| ['russian', 'covid', '19', 'vaccine', 'sputnik', 'safe', 'effective',   'says', 'rdif', 'ceo']                             	| ['russian', 'covid', '19', 'vaccin', 'sputnik', 'safe', 'effect', 'say',   'rdif', 'ceo']                               	| ['russian', 'covid', '19', 'vaccine', 'sputnik', 'safe', 'effective',   'say', 'rdif', 'ceo']                                       	|
| @Zlatty @odonnell_r @goodyear Your header   says it all ware a mask! So your asleep sheep! Also be sure to take the   vaccine! As Iâ€™m sure u always do! And hate yourself! 	| your header says it all ware mask so your   asleep sheep also be sure to take the vaccine as im sure always do and hate   yourself  	| ['header', 'says', 'ware', 'mask', 'asleep', 'sheep', 'also', 'sure',   'take', 'vaccine', 'im', 'sure', 'always', 'hate'] 	| ['header', 'says', 'ware', 'mask', 'asleep', 'sheep', 'also', 'sure',   'take', 'vaccine', 'im', 'sure', 'always', 'hate'] 	| ['header', 'say', 'ware', 'mask', 'asleep', 'sheep', 'also', 'sure',   'take', 'vaccin', 'im', 'sure', 'alway', 'hate'] 	| ['header', 'say', 'ware', 'mask', 'asleep', 'sheep', 'also', 'sure',   'take', 'vaccine', '-PRON-', 'be', 'sure', 'always', 'hate'] 	|

**Model Training**

The Doc2Vec distributed memory model is trained on the tweet lemmas using gensim, as per the documentation in <a href=" https://radimrehurek.com/gensim/models/doc2vec.html" target="_blank">Gensim.</a> 

Feature vector size of 200 is used, and each tweet is treated as its own document. The neural network is trained for 30 epochs. 

![Alt Text](https://github.com/RachelDoehr/nlp/blob/master/reports/figures/gensim_training.PNG?raw=true)

**Dimensionality Reduction**

Once the learned vector representations have been created for the covid-19 vaccine tweet vocabulary, each word can be representated as a 200x1 vector. A vector size of 200 does not lend itself well to visualization, so in order to distill the embeddings to a manageable space, we implement UMAP on the words of interest.

The words of interest are manually selected, and then built out by obtaining the 25 words most similar to each initial word.

While t-SNE is usually used for dimensionality reduction here, we use UMAP. UMAP is a new technique by McInnes et al. that offers a number of advantages over t-SNE, most notably increased speed and better preservation of the data's global structure. For example, UMAP can project the 784-dimensional, 70,000-point MNIST dataset in less than 3 minutes, compared to 45 minutes for scikit-learn's t-SNE implementation. A detailed explanation of UMAP can be found <a href="  https://pair-code.github.io/understanding-umap/" target="_blank">here,</a> including this excellent visualization of how UMAP compares to t-SNE:

![Alt Text](https://github.com/RachelDoehr/nlp/blob/master/reports/figures/umap_gif.gif?raw=true)

**Dashboard App**

To make a dynamic dashboard that allows users to visualize a comparison of two different words' 25 most similar words, <a href="  https://plotly.com/dash/" target="_blank">Dash</a> was used. Dash runs a Flask framework and allows data scientists to relatively quickly generate interactive graphs with Plotly.

Given that the entire Word2Vec model is too large to hold in memory simultaneously while running the dashboard, the embeddings for a selection of words were put through UMAP in advance, and stored in local numpy files which can be quickly loaded up in app.py.

The list of selected words generates the dropdown menu for comparison, and the selected reduced embeddings are plotted on 2 dimensions along with the 25 most similar words in the vocabulary.

![Alt Text](https://github.com/RachelDoehr/nlp/blob/master/reports/figures/umap_gif.gif?raw=true)

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p> 