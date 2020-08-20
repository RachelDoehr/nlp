
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

*RESULTS*
- DATA VISUALIZATION
- ERROR METRICS
- FEATURE IMPORTANCE

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

## Results

**Preliminary Data Visualization**

*The dataset consists of 5 continous variables and the remaining 8 are categorical or binary, which are handled appropriately with dummy variables. The target, 0 or 1, represents whether or not a patient has heart disease as indicated by contraction by >50% of any major heart vessel.*

We begin by plotting the raw distributions of the continous variables with histograms:

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/continous_variables_dist.png?raw=true)

Four of the five appear relatively normal distributions, albeit with slight skews. ST depression, however, is not. The same variables' distributions bifurcated by whether or not the patient had heart disease (using a kernel density estimator) are:

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/continous_variables_by_target.png?raw=true)

Some of those variables do appear to significantly vary by target. Additionally, the remaining categorical variables distributions include:
![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/categorical_variables_dist.png?raw=true)
 
Additional visualizations are available in /reports/figures.

**Error Metrics**

Grid search k-fold cross-validation is used across a variety of hyperparameters to select the optimal fit for each of the models examined.

| Model                               	| OOS Accuracy 	|
|-------------------------------------	|--------------	|
| Logistic Regression                 	| 88.5%        	|
| Random Forest                       	| 86.9%        	|
| AdaBoost Decision Trees             	| 83.6%        	|
| Voting Classifier of Models (1)-(3) 	| 88.5%        	|

In addition to the simple accuracy above, the confusion matrices for Logistic Regression and Random Forests' performance on the test set are (others available in /reports/figures):

*Logistic Regression*

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/conf_matrix_Logistic_Regression.png?raw=true)

*Random Forest*

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/conf_matrix_Random_Forest.png?raw=true)

We can see above that the primary difference in the errors is from the random forest over-predicting heart disease when it truly was not there (14% of patients whose true label is no disease vs. the logistic regression's 10%). This may not be a bad thing (false positives) in a clinical preventative context, however, as it is almost certainly better than false negatives.

The ROC curves illustrate this point as well, shown below. That being said, there isn't really a trade-off between false positives and false negatives between logistic regression and random forest; rather, logistic regression has lower false positives and performs the exact same as the random forest at identifying/handling instances of actually having heart disease (it does not miss them).

*Logistic Regression*

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/roc_curve_Logistic_Regression.png?raw=true)

*Random Forest*

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/roc_curve_Random_Forest.png?raw=true)

The results for the boosted decision trees and the voting classifier are available in /reports/figures, although the boosted trees underperform both simple LR and RF, while the voting classifier performs comparably, likely due to the higher weighting placed on the LR.

**Feature Importance**

We can examine the learned behavior of the trained models with respect to variables of interest. Although regularization is used in the logit model, skewing any sort of standard errors or statistical inference measures by depressing the parameter estimates (and similarly, constraining the max depth of the trees in the nonparametric estimators), it is interesting to understand the model's internal dynamics nevertheless.

I calculate what the 'average' male and female in the dataset look like by taking the mean of the continous variable by group or mode for categorical/binary variables. Next, 'synthetic' data is generated by duplicating the gender averages across the range of potential cholesterol and maximum heart rate ("thalach") values seen in the data.

Finally, these ~4,000 synthetic data points are pushed through each trained model to generate a predicted probability of heart disease. Plotting these creates a surface, over which I plotted the scatter points of the actual predicted probabilities for the ~300 patients in the dataset:

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/probability_plot_logreg%20copy.png?raw=true)

The graph above, the logit model's output, illustrates both the necessarily linear nature of the model as well as the intuitive positive link between higher maximum heart rates noted on admission to the hospital and the likelihood of cardiac disease. On the other hand, the plane is relatively invariant with respect to changes in serum cholesterol levels. 

Comparing the above to the learned decision space for the random forest highlights the nonlinearity of decision trees:

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/probability_plot_random_forest.png?raw=true)

One can immediately see how the RF is bucketing patients along different values of these continous variables, with a cluster of high risk individuals at low cholesterol + high max heart rate, and the remaining spread out below. Interestingly, the decision plane appears to put higher cholesterol levels at lower risk of cardiac disease, as long as max heart rate is not elevated.

Finally, the boosted decision trees appear somewhat similar to the results from random forest (natural), although the total range of predicted probabilities is significantly constrained relative to other models (bunched near 0.5 rather than tending toward clusters at 0 or 1). Also, the transition between 'buckets' in moving up and down the possible values of thalach and cholesterol are much more pronounced, which presumably reflects the fact that this model is a collection of decision trees with binary nodes that solve for improving the error metrics for individual datapoints, while the RF is a collection of trees over which predicted probabilities are averaged, leading to a smoother transition.

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/probability_plot_adaboost_trees.png?raw=true)

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p> 