__author__ = 'yi-linghwong'

import sys
import pandas as pd
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn import metrics
import numpy as np
import scipy as sp
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt
from collections import Counter
from nltk.util import ngrams


class ExtraTree():
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
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1,3))
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation###

        tfidf_transformer = TfidfTransformer(use_idf = True)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        # train the classifier
        clf = ExtraTreesClassifier().fit(X_tfidf, y_train)

        print ("Fitting data ...")
        clf.fit(X_tfidf, y_train)


        ##################
        # run classifier on test data
        ##################

        X_test_CV = count_vect.transform(docs_test)

        print ("Shape of test data is "+str(X_test_CV.shape))

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        scores = cross_val_score(clf, X_test_tfidf, y_test, cv=3, scoring='f1_weighted')
        print ("Cross validation score:%s " % scores)

        y_predicted = clf.predict(X_test_tfidf)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score is: %s " % clf.score(X_test_tfidf,y_test))

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf,count_vect


    def train_classifier_use_feature_selection(self):

        # Get list of features
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1,3))
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation###

        tfidf_transformer = TfidfTransformer(use_idf = False)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        selector = SelectPercentile(score_func=chi2, percentile=85)

        print ("Fitting data with feature selection ...")
        selector.fit(X_tfidf, y_train)

        # get how many features are left after feature selection
        X_features = selector.transform(X_tfidf)

        print ("Shape of array after feature selection is "+str(X_features.shape))

        clf = ExtraTreesClassifier().fit(X_features, y_train)

        ####################
        #test clf on test data
        ####################

        X_test_CV = count_vect.transform(docs_test)

        print ("Shape of test data is "+str(X_test_CV.shape))

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        # apply feature selection on test data too
        X_test_selector = selector.transform(X_test_tfidf)
        print ("Shape of array for test data after feature selection is "+str(X_test_selector.shape))

        y_predicted = clf.predict(X_test_selector)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score is: %s " % clf.score(X_test_selector,y_test))

        # returns cross validation score

        scores = cross_val_score(clf, X_features, y_train, cv=3, scoring='f1_weighted')
        print ("Cross validation score:%s " % scores)


        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf,count_vect


    def use_pipeline(self):

        #####################
        #Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([
                ('vect', TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=0.90)),
                ('clf', ExtraTreesClassifier()),
        ])


        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'vect__ngram_range': [(1,2), (1,3)],
            'vect__use_idf': (True, False),
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

        #y_predicted = clf_gs.predict(docs_test)


        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        ngram_range = best_parameters['vect__ngram_range']
        use_idf = best_parameters['vect__use_idf']

        # vectorisation

        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation

        tfidf_transformer = TfidfTransformer(use_idf=use_idf)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        # train the classifier

        clf = ExtraTreesClassifier().fit(X_tfidf, y_train)

        print ("Fitting data with best parameters ...")
        clf.fit(X_tfidf, y_train)

        # run classifier on test data

        X_test_CV = count_vect.transform(docs_test)

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        scores = cross_val_score(clf, X_test_tfidf, y_test, cv=3, scoring='f1_weighted')
        print ("Cross validation score:%s " % scores)

        y_predicted = clf.predict(X_test_tfidf)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score is: %s " % clf.score(X_test_tfidf,y_test))


        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        # import matplotlib.pyplot as plt
        # plt.matshow(cm)
        # plt.show()

        return clf,count_vect


    def use_pipeline_with_fs(self):

        #####################
        #Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([
                ('vect', TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=0.90)),
                ("selector", SelectPercentile()),
                ('clf', ExtraTreesClassifier()),
        ])


        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'vect__ngram_range': [(1,2), (1,3)],
            'vect__use_idf': (True, False),
            'selector__score_func': (chi2, f_classif),
            'selector__percentile': (85, 95),
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

        #y_predicted = clf_gs.predict(docs_test)

        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        ngram_range = best_parameters['vect__ngram_range']
        use_idf = best_parameters['vect__use_idf']
        score_func = best_parameters['selector__score_func']
        percentile = best_parameters['selector__percentile']

        # vectorisation

        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation

        tfidf_transformer = TfidfTransformer(use_idf=use_idf)

        selector = SelectPercentile(score_func=score_func, percentile=percentile)

        combined_features = Pipeline([
            ("vect", count_vect),
            ("tfidf", tfidf_transformer),
            ("feat_select", selector)
        ])

        X_features = combined_features.fit_transform(docs_train,y_train)
        X_test_features = combined_features.transform(docs_test)

        print ("Shape of train data after feature selection is "+str(X_features.shape))
        print ("Shape of test data after feature selection is "+str(X_test_features.shape))


        # run classifier on selected features

        clf = ExtraTreesClassifier().fit(X_features, y_train)

        y_predicted = clf.predict(X_test_features)


        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        # import matplotlib.pyplot as plt
        # plt.matshow(cm)
        # plt.show()

        return clf,count_vect


    def get_important_features(self, clf, count_vect):

        # get vocabulary
        vocab = count_vect.vocabulary_
        vl = []

        for key in vocab.keys():
            vl.append(key)

        vl.sort()

        f = open(path_to_store_vocabulary_file, 'w')

        for v in vl:
            f.write(str(v) + '\n')
        f.close()


        # get most important features
        feat_imp = clf.feature_importances_

        f = open(path_to_store_complete_feature_importance_file, 'w')
        for fea in feat_imp:
            f.write(str(fea) + '\n')
        f.close()

        lines = open(path_to_store_complete_feature_importance_file, 'r').readlines()
        lines2 = open(path_to_store_vocabulary_file, 'r').readlines()

        # create a list with elements sorted by feature importance
        sortli = sorted(range(len(feat_imp)), key=lambda i: feat_imp[i], reverse=True)[:100]

        f = open(path_to_store_top_important_features_file, 'w')
        for i in sortli:
            f.write(lines[i].replace('\n', '') + '\t' + lines2[i].replace('\n', '') + '\n')
        f.close()


        ##################
        # get feature importance by class
        ##################

        lines = open(path_to_labelled_file, 'r').readlines()

        l1=[] #hrt ngram list
        l2=[] #lrt ngram list

        for line in lines:
            spline=line.replace("\n", "").split(",")
            #creates a list with key and value. Split splits a string at the comma and stores the result in a list

            #create a list of word which includes ngrams
            n=4

            if spline[1] == 'HRT':
                for i in range(1,n):
                    n_grams = ngrams(spline[0].split(), i) #output [('one', 'two'), ('two', 'three'), ('three', 'four')]
                    #join the elements within the list together
                    gramify = [' '.join(x) for x in n_grams] #output ['one two', 'two three', 'three four']
                    l1.extend(gramify)

            elif spline[1] == 'LRT':
                for i in range(1,n):
                    n_grams = ngrams(spline[0].split(), i)
                    gramify = [' '.join(x) for x in n_grams]
                    l2.extend(gramify)

        ##combine lists in a list into one long list. Flattening a list.
        #hrt_t = list(itertools.chain(*l1))
        #lrt_t = list(itertools.chain(*l2))

        hrt = [w.lower() for w in l1]
        lrt = [w.lower() for w in l2]

        lines2 = open(path_to_store_top_important_features_file, 'r').readlines()

        features=[]

        for line in lines2:
            spline=line.replace("\n", "").split("\t")

            features.append(spline[1])


        feat_by_class=[]

        for f in features:

            # count the occurrences of each important feature in HRT and LRT list respectively
            hrt_count = hrt.count(f)
            lrt_count = lrt.count(f)

            print ("HRT %s: " % f + str(hrt_count))
            print ("LRT %s: " % f + str(lrt_count))

            if (hrt_count-lrt_count)>10:

                feat_by_class.append('HRT'+','+f+','+str(hrt_count))


            elif (lrt_count-hrt_count)>10:
                feat_by_class.append('LRT'+','+f+','+str(lrt_count))

            else:
                feat_by_class.append('BOTH'+','+f+','+str(hrt_count)+','+str(lrt_count))


        feat_by_class = sorted(feat_by_class)

        file = open(path_to_store_important_features_by_class_file, 'w')

        for f in feat_by_class:
            file.write(f+'\n')
        file.close()


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

        clf = Pipeline([('anova', transform), ('clf', ExtraTreesClassifier())])

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
            'Performance of the ExtraTree-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')

        plt.axis('tight')
        plt.show()


###############
# variables
###############

path_to_labelled_file = 'test.txt'
path_to_stopword_file = '../../TwitterML/stopwords/stopwords.csv'
path_to_store_vocabulary_file = '../output/feature_importance/extratree_vocab.txt'
path_to_store_complete_feature_importance_file = '../output/feature_importance/extratree_feat_imp_all.txt'
path_to_store_top_important_features_file = '../output/feature_importance/extratree_feature_importance.csv'
path_to_store_important_features_by_class_file = '../output/feature_importance/extratree_feat_byClass.csv'


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

    et = ExtraTree()

    ###################
    # select one of the method to split data using Cross Validation
    ###################

    docs_train,docs_test,y_train,y_test = et.train_test_split()
    #docs_train,docs_test,y_train,y_test = et.stratified_shufflesplit()
    #docs_train,docs_test,y_train,y_test = et.stratified_kfolds()


    ##################
    # run ExtraTree Classifier
    ##################

    #clf, count_vect = et.train_classifier()


    ###################
    # run ExtraTree Classifier and use feature selection
    ###################

    #clf, count_vect = et.train_classifier_use_feature_selection()


    ###################
    # use pipeline
    ###################

    #clf, count_vect = et.use_pipeline()

    ###################
    # use pipeline and use feature selection
    ###################

    clf, count_vect = et.use_pipeline_with_fs()


    ###################
    # Get feature importance
    ###################

    et.get_important_features(clf,count_vect)


    ##################
    # Plot feature selection
    ##################

    #et.plot_feature_selection()
