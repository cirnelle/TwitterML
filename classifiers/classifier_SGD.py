__author__ = 'yi-linghwong'

import sys
import os
import pandas as pd
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn import metrics
import scipy as sp
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt
import numpy as np


class SGD():
    def train_test_split(self):

        #################
        # Split the dataset in training and test set:
        #################

        docs_train, docs_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        print("Number of data point is " + str(len(y)))

        return docs_train, docs_test, y_train, y_test

    def stratified_shufflesplit(self):

        ####################
        # Stratified ShuffleSplit cross validation iterator
        # Provides train/test indices to split data in train test sets.
        # This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds.
        # The folds are made by preserving the percentage of samples for each class.
        ####################


        sss = StratifiedShuffleSplit(y, 5, test_size=0.2, random_state=42)

        for train_index, test_index in sss:
            print("TRAIN:", train_index, "TEST:", test_index)
            docs_train, docs_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return docs_train, docs_test, y_train, y_test

    def stratified_kfolds(self):

        ##################
        # Stratified K-Folds cross validation iterator
        # Provides train/test indices to split data in train test sets.
        # This cross-validation object is a variation of KFold that returns stratified folds.
        # The folds are made by preserving the percentage of samples for each class.
        ##################

        skf = StratifiedKFold(y, n_folds=5)

        print(len(skf))
        print(skf)

        for train_index, test_index in skf:
            print("TRAIN:", train_index, "TEST:", test_index)
            docs_train, docs_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return docs_train, docs_test, y_train, y_test

    def train_classifier(self):

        # Get list of features
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1, 3))
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print("Shape of train data is " + str(X_CV.shape))

        # tfidf transformation###

        tfidf_transformer = TfidfTransformer(use_idf=True)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        # train the classifier
        clf = SGDClassifier(loss='hinge', penalty='elasticnet', random_state=42).fit(X_tfidf, y_train)

        print("Fitting data ...")
        clf.fit(X_tfidf, y_train)

        ##################
        # run classifier on test data
        ##################

        X_test_CV = count_vect.transform(docs_test)

        print("Shape of test data is " + str(X_test_CV.shape))

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        scores = cross_val_score(clf, X_test_tfidf, y_test, cv=3, scoring='f1_weighted')
        print("Cross validation score:%s " % scores)

        y_predicted = clf.predict(X_test_tfidf)

        # print the mean accuracy on the given test data and labels

        print("Classifier score is: %s " % clf.score(X_test_tfidf, y_test))

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf, count_vect

    def train_classifier_use_feature_selection(self):

        # Get list of features
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1, 3))
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print("Shape of train data is " + str(X_CV.shape))

        # tfidf transformation###

        tfidf_transformer = TfidfTransformer(use_idf=True)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        selector = SelectPercentile(score_func=chi2, percentile=95)

        print("Fitting data with feature selection ...")
        selector.fit(X_tfidf, y_train)

        # get how many features are left after feature selection
        X_features = selector.transform(X_tfidf)

        print("Shape of array after feature selection is " + str(X_features.shape))

        clf = SGDClassifier(loss='hinge', penalty='elasticnet', random_state=42).fit(X_features, y_train)

        ####################
        # test clf on test data
        ####################

        X_test_CV = count_vect.transform(docs_test)

        print("Shape of test data is " + str(X_test_CV.shape))

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        # apply feature selection on test data too
        X_test_selector = selector.transform(X_test_tfidf)
        print("Shape of array for test data after feature selection is " + str(X_test_selector.shape))

        y_predicted = clf.predict(X_test_selector)

        # print the mean accuracy on the given test data and labels

        print("Classifier score is: %s " % clf.score(X_test_selector, y_test))

        # returns cross validation score

        scores = cross_val_score(clf, X_features, y_train, cv=3, scoring='f1_weighted')
        print("Cross validation score:%s " % scores)

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf, count_vect

    def use_pipeline(self):

        #####################
        # Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([
            ('vect', TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=0.90)),
            # ('clf', LinearSVC(C=1000)),
            ('clf', SGDClassifier(random_state=42))
        ])

        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'vect__ngram_range': [(1, 2), (1, 3)],
            'vect__use_idf': (True, False),
            'clf__loss': ('hinge', 'log'),
            'clf__penalty': ('none', 'l1', 'elasticnet'),
            'clf__alpha': (0.0001, 0.0005),
        }

        cv = StratifiedShuffleSplit(y_train, n_iter=5, test_size=0.2, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=cv, n_jobs=-1)
        clf_gs = grid_search.fit(docs_train, y_train)


        ###############
        # print the cross-validated scores for the each parameters set explored by the grid search
        ###############


        best_parameters, score, _ = max(clf_gs.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        print("score is %s" % score)

        # y_predicted = clf_gs.predict(docs_test)

        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        ngram_range = best_parameters['vect__ngram_range']
        use_idf = best_parameters['vect__use_idf']
        loss = best_parameters['clf__loss']
        penalty = best_parameters['clf__penalty']
        alpha = best_parameters['clf__alpha']

        # vectorisation

        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print("Shape of train data is " + str(X_CV.shape))

        # tfidf transformation

        tfidf_transformer = TfidfTransformer(use_idf=use_idf)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        # train the classifier

        clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, random_state=42).fit(X_tfidf, y_train)

        print("Fitting data with best parameters ...")
        clf.fit(X_tfidf, y_train)

        # run classifier on test data

        X_test_CV = count_vect.transform(docs_test)

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        scores = cross_val_score(clf, X_test_tfidf, y_test, cv=3, scoring='f1_weighted')
        print("Cross validation score:%s " % scores)

        y_predicted = clf.predict(X_test_tfidf)

        # print the mean accuracy on the given test data and labels

        print("Classifier score is: %s " % clf.score(X_test_tfidf, y_test))


        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        # import matplotlib.pyplot as plt
        # plt.matshow(cm)
        # plt.show()

        return clf, count_vect

    def use_pipeline_with_fs(self):

        #####################
        # Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([
            ('vect', TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=0.90)),
            ("selector", SelectPercentile()),
            # ('clf', LinearSVC(C=1000)),
            ('clf', SGDClassifier(random_state=42))
        ])

        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'vect__ngram_range': [(1, 2), (1, 3)],
            'vect__use_idf': (True, False),
            'selector__score_func': (chi2, f_classif),
            'selector__percentile': (85, 95),
            'clf__loss': ('hinge', 'log'),
            'clf__penalty': ('none', 'l1', 'elasticnet'),
            'clf__alpha': (0.0001, 0.0005),

        }

        cv = StratifiedShuffleSplit(y_train, n_iter=5, test_size=0.2, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=cv, n_jobs=-1)
        clf_gs = grid_search.fit(docs_train, y_train)

        ###############
        # print the cross-validated scores for the each parameters set explored by the grid search
        ###############

        best_parameters, score, _ = max(clf_gs.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        print("score is %s" % score)

        # y_predicted = clf_gs.predict(docs_test)

        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        ngram_range = best_parameters['vect__ngram_range']
        use_idf = best_parameters['vect__use_idf']
        score_func = best_parameters['selector__score_func']
        percentile = best_parameters['selector__percentile']
        loss = best_parameters['clf__loss']
        penalty = best_parameters['clf__penalty']
        alpha = best_parameters['clf__alpha']

        # vectorisation

        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print("Shape of train data is " + str(X_CV.shape))

        # tfidf transformation

        tfidf_transformer = TfidfTransformer(use_idf=use_idf)

        selector = SelectPercentile(score_func=score_func, percentile=percentile)

        combined_features = Pipeline([
            ("vect", count_vect),
            ("tfidf", tfidf_transformer),
            ("feat_select", selector)
        ])

        X_features = combined_features.fit_transform(docs_train, y_train)
        X_test_features = combined_features.transform(docs_test)

        print("Shape of train data after feature selection is " + str(X_features.shape))
        print("Shape of test data after feature selection is " + str(X_test_features.shape))

        # run classifier on selected features

        clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, random_state=42).fit(X_features, y_train)

        y_predicted = clf.predict(X_test_features)


        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        # import matplotlib.pyplot as plt
        # plt.matshow(cm)
        # plt.show()

        return clf, count_vect


    def get_important_features(self,clf,count_vect):

        # get coef_

        f = open(path_to_store_coefficient_file, 'w')

        coef = clf.coef_[0]

        for c in coef:
            f.write(str(c) + '\n')

        f.close()

        # get feature names

        feature_names = count_vect.get_feature_names()

        f = open(path_to_store_list_of_feature_file, 'w')

        for fn in feature_names:
            f.write(str(fn) + '\n')
        f.close()

        # sort feature importance

        coef_list = []
        for c in coef:
            coef_list.append(c)

        feat_list = list(zip(coef_list, feature_names))

        feat_list.sort()

        f = open(path_to_store_feature_and_coef_file, 'w')

        # f.write(str(feat_list))

        for fl in feat_list:
            f.write(str(fl) + '\n')
        f.close()

        hrt = []
        lrt = []

        for fl in feat_list:
            if fl[0] < 0:
                hrt.append('HRT ' + str(fl))

            if fl[0] > 0:
                lrt.append('LRT ' + str(fl))

        f = open(path_to_store_important_features_by_class_file, 'w')

        for feat in hrt[:100]:
            f.write(str(feat) + '\n')

        for feat in reversed(lrt[-100:]):
            f.write(str(feat) + '\n')

        f.close()


    def plot_feature_selection(self):

        # vectorisation

        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1,3))
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation

        tfidf_transformer = TfidfTransformer(use_idf=True)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)


        transform = SelectPercentile(score_func=chi2)

        clf = Pipeline([('anova', transform), ('clf', SGDClassifier(loss='hinge', penalty='elasticnet', random_state=42))])

        ###############################################################################
        # Plot the cross-validation score as a function of percentile of features
        score_means = list()
        score_stds = list()
        percentiles = (10, 20, 30, 40, 60, 80, 85, 95, 100)

        for percentile in percentiles:
            clf.set_params(anova__percentile=percentile)
            # Compute cross-validation score using all CPUs
            this_scores = cross_validation.cross_val_score(clf, X_tfidf, y_train, n_jobs=-1)
            score_means.append(this_scores.mean())
            score_stds.append(this_scores.std())

        plt.errorbar(percentiles, score_means, np.array(score_stds))

        plt.title(
            'Performance of the SVM-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')

        plt.axis('tight')
        plt.show()


###############
# variables
###############

path_to_labelled_file = 'test.txt'
path_to_stopword_file = '../../TwitterML/stopwords/stopwords.csv'
path_to_store_coefficient_file = '../output/feature_importance/sgd_coef.csv'
path_to_store_list_of_feature_file = '../output/feature_importance/sgd_feature_names.csv'
path_to_store_feature_and_coef_file = '../output/feature_importance/sgd_coef_and_feat.csv'
path_to_store_important_features_by_class_file = '../output/feature_importance/sgd_feat_by_class.csv'



def get_data_set():

    #############
    # Get dataset
    #############

    dataset = pd.read_csv(path_to_labelled_file, header=0, names=['tweets', 'class'])

    X = dataset['tweets']
    y = dataset['class']

    return X,y

def get_stop_words():

    ###########
    # get stopwords
    ###########

    lines = open(path_to_stopword_file, 'r').readlines()

    my_stopwords=[]
    for line in lines:
        my_stopwords.append(line.replace("\n", ""))

    stopwords = text.ENGLISH_STOP_WORDS.union(my_stopwords)

    return stopwords


if __name__ == '__main__':

    X = get_data_set()[0]
    y = get_data_set()[1]
    stopwords = get_stop_words()

    sgd = SGD()

    ###################
    # select one of the method to split data using Cross Validation
    ###################

    docs_train,docs_test,y_train,y_test = sgd.train_test_split()
    #docs_train,docs_test,y_train,y_test = sgd.stratified_shufflesplit()
    #docs_train,docs_test,y_train,y_test = sgd.stratified_kfolds()

    ##################
    # run SGD Classifier
    ##################

    #clf, count_vect = sgd.train_classifier()


    ###################
    # run SGD Classifier and use feature selection
    ###################

    #clf, count_vect = sgd.train_classifier_use_feature_selection()


    ###################
    # use pipeline
    ###################

    #clf, count_vect = sgd.use_pipeline()

    ###################
    # use pipeline and use feature selection
    ###################

    clf, count_vect = sgd.use_pipeline_with_fs()


    ###################
    # Get feature importance
    ###################

    #sgd.get_important_features(clf,count_vect)


    ##################
    # Plot feature selection
    ##################

    #sgd.plot_feature_selection()
