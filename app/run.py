import json
import plotly
import numpy as np
import pandas as pd
from collections import Counter
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    """Tokenize a text meaning
    bag of word, remove punctuation and stop
    words, make it lower and lemmatize

    :param text: text to tokenize
    :return: clean toknized text
    """
    text = text.lower().strip()
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    word_text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(word_text)
    word_tokens = [word for word in tokens if
                   word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in word_tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def most_common(messages, load=True, filepath='../data/word.npz'):
    """
    Compute most common words(10)
    :param messages: messages to process
    :param load: if we need to load existing data
    :param filepath: path to save and load data
    :return:
    """
    if load:
        data = np.load(filepath)
        return list(data['top_words']), list(data['top_counts'])
    else:
        counter = Counter()
        for message in messages:
            tokens = tokenize(message)
            for token in tokens:
                counter[token] += 1
        top10 = counter.most_common(10)
        top_words = [word[0] for word in top10]
        top_counts = [count[1] for count in top10]
        np.savez(filepath, top_words=top_words, top_counts=top_counts)
        return list(top_words), list(top_counts)


def lsa(messages, load=True, filepath='../data/lsa.npz'):
    """
    Compute lsa algorithms to plot word similarity
    :param messages: messages text to compute
    :param load: bool to indicate weither data is loaded
    :param filepath: path to save or load data
    :return: tuple containing
    first and second component of SVD
    and features names
    """

    if load:
        data = np.load(filepath)
        return list(data['Z0']), list(data['Z1']), list(data['names'])
    else:
        vect = TfidfVectorizer(max_features=100,
                               tokenizer=tokenize)
        X_tfidf = vect.fit_transform(messages)
        # SVD
        svd = TruncatedSVD()
        Z = svd.fit_transform(X_tfidf.toarray().T)
        # save arrays
        np.savez(filepath, Z0=Z[:,0], Z1=Z[:,1],
                 names=vect.get_feature_names())
        return Z[:,0], Z[:,1], vect.get_feature_names()


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Create new transform which check
     it a word is a starting verb
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) > 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # target distribution
    target_distribution = df.drop(['id', 'message', 'original', 'genre'],
                                  axis=1).mean()
    target_names = list(target_distribution.index)

    # top 10 words and counts
    top_words, top_counts = most_common(df.message.values)
    # LSA
    Z0, Z1, names = lsa(df.message.values)
    # create grahs
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=target_names,
                    y=target_distribution
                )
            ],

            'layout': {
                'title': 'Percentage of Needs',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Needs"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_counts
                )
            ],

            'layout': {
                'title': 'Top 10 words and counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=Z0,
                    y=Z1,
                    mode='markers+text',
                    name='Markers and Text',
                    text=names,
                    textposition='top center'
                )
            ],

            'layout': {
                'title': 'Word Similarities (LSA)',
                'xaxis': dict(
                    autorange=True,
                    zeroline=False,
                    ticks='',
                    showticklabels=False
                ),
                'yaxis': dict(
                    autorange=True,
                    zeroline=False,
                    ticks='',
                    showticklabels=False
                )
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    # render web page with plotly graphs
    #print(len(ids))
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()