__author__ = 'yi-linghwong'

import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import scipy as sp


if __name__ == "__main__":

    dataset = pd.read_csv('output/output_engrate_label_080815_noART.csv', header=0, names=['tweets', 'class'])

    X = dataset['tweets']
    y = dataset['class']

    print (len(X))

    # Split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    '''

    ### Get list of features
    count_vect = CountVectorizer(stop_words='english', min_df=3, max_df=0.95)
    X_CV = count_vect.fit_transform(docs_train)
    ### print number of unique words (n_features)
    print (X_CV.shape)

    ### Get list of features
    #sys.stdout = open('output/output_features.txt', "w")
    #print (count_vect.vocabulary_)


    ###tfidf transformation
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_CV)
    print(X_tfidf.shape)

    ###training the classifier
    clf = MultinomialNB().fit(X_tfidf, y_train)

    print (clf.class_count_)

    ##test clf on test data

    X_test_CV = count_vect.transform(docs_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_CV)

    y_predicted = clf.predict(X_test_tfidf)

    # Print and plot the confusion matrix

    print(metrics.classification_report(y_test, y_predicted))
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)


    '''

    # Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
    pipeline = Pipeline([
            ('vect', TfidfVectorizer(stop_words='english', min_df=3, max_df=0.90)),
            #('clf', LinearSVC(C=1000)),
            ('clf', SGDClassifier(loss='hinge', penalty='l2', n_iter=5, random_state=42))

    ])


    # Build a grid search to find the best parameter
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
        'vect__ngram_range': [(1, 2), (1, 3)],
        'vect__use_idf': (True, False)
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    clf_gs = grid_search.fit(docs_train, y_train)


    # print the cross-validated scores for the each parameters set
    # explored by the grid search
    #print(clf_gs.grid_scores_)


    best_parameters, score, _ = max(clf_gs.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    print("score is %s" % score)


    y_predicted = clf_gs.predict(docs_test)

    # Print and plot the confusion matrix

    print(metrics.classification_report(y_test, y_predicted))
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.show()





