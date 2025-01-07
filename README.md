**Sentiment Analysis and Topic Modeling**
This project performs sentiment analysis on a text dataset, generates word clouds based on sentiment, and builds both a Logistic Regression and LSTM model for sentiment classification. Additionally, it uses Latent Dirichlet Allocation (LDA) for topic modeling.

USED LIBRARIES AND FUNCTIONALITIES:
pandas: Provides DataFrames for efficient data manipulation and analysis.
numpy: Supports numerical operations on arrays and matrices.
matplotlib.pyplot: Enables creation of static visualizations in Python.
seaborn: Builds on matplotlib to simplify statistical graphic creation.
wordcloud: Generates word cloud visualizations from textual data.
nltk: Offers tools for natural language processing (NLP) tasks.
nltk.corpus: Provides access to various text corpora.
nltk.tokenize: Implements methods for splitting text into words or sentences.
nltk.stem: Implements stemming and lemmatization for text normalization.
nltk.tokenize.toktok: A fast tokenizer for NLP tasks.
re: Enables regular expressions for pattern matching in strings.
sklearn.model_selection: Tools for splitting data into training and test sets.
sklearn.feature_extraction.text: Provides methods for text feature extraction (e.g., TF-IDF, CountVectorizer).
sklearn.preprocessing: Includes techniques for data preprocessing.
sklearn.linear_model: Implements linear models like Logistic Regression.
sklearn.metrics: Provides methods for evaluating model performance.
sklearn.naive_bayes: Implements Naive Bayes for classification tasks.
PROJECT OVERVIEW:
This project performs sentiment analysis, word cloud generation, and topic modeling using text data. The main features of the project are:

Sentiment Analysis:
Builds and evaluates sentiment classification models using both Logistic Regression and LSTM.
Word Cloud Generation:
Generates word clouds for both positive and negative reviews using the wordcloud library.
Topic Modeling:
Uses LDA (Latent Dirichlet Allocation) to identify latent topics within the reviews.
DATA EXPLORATION AND PREPROCESSING:
Data Visualization:
Sentiment distribution visualized using sns.countplot.
Word length distribution analyzed for positive and negative reviews.
Word clouds generated for positive and negative sentiment using WordCloud.
Text Preprocessing:
HTML tags, special characters, and stop words are removed from the reviews.
The dataset is split into training and testing sets.
FEATURE EXTRACTION:
CountVectorizer (Bag-of-Words) and TfidfVectorizer are used to convert the textual reviews into numerical features based on word frequency and importance.
SENTIMENT CLASSIFICATION MODELS:
Logistic Regression:

Two Logistic Regression models are trained: one using Bag-of-Words (BOW) features and another using TF-IDF features.
Performance of the models is evaluated using classification metrics like precision, recall, and F1-score.
LSTM Model:

An LSTM model is built and trained for sentiment classification.
The training and validation accuracy of the LSTM model are visualized during training.
LDA TOPIC MODELING:
Text data is preprocessed (including stemming and stop word removal) for topic modeling.
The reviews are converted into a bag-of-words representation.
An LDA model is built to identify latent topics within the reviews.
The most frequent words associated with each topic are displayed.
