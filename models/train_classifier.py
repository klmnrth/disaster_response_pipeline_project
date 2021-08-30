# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import pickle


def load_data(database_filepath):
    """
    Load Data

    This funtion is loading the data from the database file (.db) and is defining feature and target variables X and Y

    Args: Filepath to database file (.db)
    """
    
    #Load data from database
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df_table', engine)

    #Define feature and target variables X and Y
    X = df['message'].values
    y = df[df.columns[4:]].values
    category_names = df.columns[4:]

    return X, y, category_names


def tokenize(text):
    """
    Tokenize function

    This function is tokenizing the provided text. The following steps are processed:
        - Normalize text
        - Tokenize text
        - Remove stopwords
        - Lemmatize


    Args:   Text
    Output: Tokenized text
    """
    #Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    
    #Tokenize
    tokens = word_tokenize(text)
    
    #Remove Stop Words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    #Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build Model

    This function is initiating the pipeline which is building the model.
    Output: ML model
    """

    # build pipeline
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model

    This function is evaluating the model by predicting on test data and printing the classification report.

    Args:   ML model
            X_test (features)
            y_test (labels)
            category_names (label names)
    """
    y_pred = model.predict(X_test)

    for c in range(0, len(category_names)):
        print(category_names[c] + ":")
        print(classification_report(Y_test[:,c], y_pred[:,c], zero_division = 0))


def save_model(model, model_filepath):
    """
    Save model

    This function is saving the trained ML model to the provided destination path as a pickle file (.pkl).

    Args:   ML model 
            model_filepath (destination path)
    """
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main Function

    This is the Main Function which is processing the steps of the ML pipeline.
        - Load Data from database file (.db)
        - Build the pipeline / the model
        - Train the model
        - Evaluate the model
        - Saves the model as a pickle file (.pkl)
    """
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