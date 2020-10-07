import sys
import numpy as np
import pandas as pd
import re
import pickle


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report

nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])


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


def load_data(database_filepath):
    """Load data from database file
    :param database_filepath: path to database
    :return: tuple of X, y matrix and categories
    name
    meaning X is matrix of features and y
    labels, categories name
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("DisasterTable", con=engine)
    X = df.message
    y = df.iloc[:, 4:]
    cat_names = y.columns.tolist()
    return X, y, cat_names


def tokenize(text):
    """Tokenize a text meaning
    bag of word, remove punctuation and stop
    words, make it lower and lemmatize

    :param text: text to tokenize
    :return: clean toknized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]' + \
                 '|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build Adabost model by using pipeline
    and Gridsearch
    :return:
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('aclf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {
        'aclf__estimator__learning_rate': [0.1, 0.2, 0.3],
        'aclf__estimator__n_estimators': [100, 200]
    }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3,
                      verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the trainedmodel
    :param model: estimator
    :param X_test: disaster message to test
    :param Y_test: disaster label
    :param category_names: categories of the label
    :return: print classification of each  categories
    """
    y_prediction = model.predict(X_test)
    print(classification_report(Y_test.iloc[:, 1:].values,
                                np.array([x[1:] for x in y_prediction]),
                                target_names=category_names))


def save_model(model, model_filepath):
    """Save the model with pickle
    :param model: estimator
    :param model_filepath: path file to save the model
    :return: None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()