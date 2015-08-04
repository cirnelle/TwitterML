__author__ = 'yi-linghwong'

import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics


if __name__ == "__main__":

    dataset = pd.read_csv('output/output_engrate_label.csv', header=0, names=['tweets', 'class'])

    X = dataset['tweets']
    y = dataset['class']

    print (len(X))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

    pipeline = Pipeline([
            ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
            ('clf', MultinomialNB()),
    ])
