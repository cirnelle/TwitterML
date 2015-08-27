__author__ = 'yi-linghwong'

import sys
import pandas as pd
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.svm import LinearSVC
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


class SVM():


    def train_test_split(self):

        """Split the dataset in training and test set:"""
        docs_train, docs_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        return docs_train, docs_test, y_train, y_test


    def stratified_shufflesplit(self):


        """Stratified ShuffleSplit cross validation iterator"""
        #Provides train/test indices to split data in train test sets.
        #This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds.
        #The folds are made by preserving the percentage of samples for each class.


        sss = StratifiedShuffleSplit(y, 5, test_size=0.2, random_state=42)

        print (len(sss))
        print (sss)

        for train_index, test_index in sss:
            print("TRAIN:", train_index, "TEST:", test_index)
            docs_train, docs_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return docs_train, docs_test, y_train, y_test

    def stratified_kfolds(self):


        """Stratified K-Folds cross validation iterator"""
        #Provides train/test indices to split data in train test sets.
        #This cross-validation object is a variation of KFold that returns stratified folds.
        # The folds are made by preserving the percentage of samples for each class.


        skf = StratifiedKFold(y, n_folds=5)

        print (len(skf))
        print(skf)

        for train_index, test_index in skf:
            print("TRAIN:", train_index, "TEST:", test_index)
            docs_train, docs_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return docs_train, docs_test, y_train, y_test


    def get_features(self, docs_train):


        ### Get list of features
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1,3))
        X_CV = count_vect.fit_transform(docs_train)
        ### print number of unique words (n_features)
        print (X_CV.shape)

        ### Get list of features ###
        #file = open('output/output_features.txt', "w")
        #file.write(count_vect.get_feature_names())
        #file.write(count_vect.vocabulary_)
        #print (len(count_vect.vocabulary_))
        #file.close()

        ###tfidf transformation###

        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        return X_tfidf, count_vect, tfidf_transformer

    def train_classifier(self,X_tfidf,y_train,docs_test,y_test,count_vect,tfidf_transformer):

        ###training the classifier
        clf = SGDClassifier(loss='hinge', penalty='l2', n_iter=5, random_state=42).fit(X_tfidf, y_train)

        ##test clf on test data

        X_test_CV = count_vect.transform(docs_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        scores = cross_val_score(clf, X_test_tfidf, y_test, cv=3, scoring='f1_weighted')
        print ("Cross validation score:%s " % scores)

        y_predicted = clf.predict(X_test_tfidf)

        #print the mean accuracy on the given test data and labels.
        print ("Classifier score is: %s " % clf.score(X_test_tfidf,y_test))

        return y_predicted, clf

    def use_feature_selection(self,X_tfidf,y_train,docs_test,y_test,count_vect,tfidf_transformer):

        ## create the classifier object and use feature selection ###

        svm = SGDClassifier(loss='hinge', penalty='l2', n_iter=3, random_state=42)

        # The "accuracy" scoring is proportional to the number of correct
        # classifications

        #rfecv = RFE(estimator=svm, n_features_to_select=10000, step=1) #use recursive feature elimination
        #rfecv = RFECV(estimator=svm, step=1, cv=3, scoring='accuracy')#use recursive feature elimination with cross validation
        selector = SelectPercentile(score_func=chi2, percentile=85)

        print ("Fit transform using feature selection ...")

        selector.fit(X_tfidf, y_train)
        X_features = selector.transform(X_tfidf)
        print (X_features.shape)

        #print("Optimal number of features : %d" % rfecv.n_features_)

        clf = SGDClassifier(loss='hinge', penalty='l2', n_iter=5, random_state=42).fit(X_features, y_train)

        """test clf on test data"""


        X_test_CV = count_vect.transform(docs_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_CV)
        X_test_selector = selector.transform(X_test_tfidf)
        print (X_test_selector.shape)

        y_predicted = clf.predict(X_test_selector)

        """print the mean accuracy on the given test data and labels"""

        print ("Classifier score is: %s " % clf.score(X_test_selector,y_test))

        """returns cross validation score"""

        scores = cross_val_score(clf, X_features, y_train, cv=3, scoring='f1_weighted')
        print ("Cross validation score:%s " % scores)

        return y_predicted, clf


    def confusion_matrix(self,y_test,y_predicted):

        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

    def get_important_features(self,clf,count_vect):

        ###get coef_ ###

        f=open('output/feature_importance/svm_coef.csv', 'w')

        coef=clf.coef_[0]

        for c in coef:

            f.write(str(c)+'\n')

        f.close()

        ###get feature names###

        feature_names=count_vect.get_feature_names()

        f=open('output/feature_importance/svm_feature_names.csv', 'w')

        for fn in feature_names:
            f.write(str(fn)+'\n')
        f.close()

        ###sort feature importance###

        coef_list=[]
        for c in coef:
            coef_list.append(c)

        feat_list=list(zip(coef_list,feature_names))

        feat_list.sort()


        f=open('output/feature_importance/svm_coef_and_feat.csv', 'w')

        #f.write(str(feat_list))

        for fl in feat_list:
            f.write(str(fl)+'\n')
        f.close()

        hrt=[]
        lrt=[]

        for fl in feat_list:
            if fl[0] < 0:
                hrt.append('HRT '+str(fl))

            if fl[0] > 0:
                lrt.append('LRT '+str(fl))

        f=open('output/feature_importance/svm_feat_by_class.svm', 'w')

        for feat in hrt[:100]:
            f.write(str(feat)+'\n')

        for feat in reversed(lrt[-100:]):
            f.write(str(feat)+'\n')

        f.close()


    def use_pipeline(self,docs_train,y_train,docs_test,y_test):


        # Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        pipeline = Pipeline([
                ('vect', TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=0.90)),
                ('selector', SelectPercentile()),
                #('clf', LinearSVC(C=1000)),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', n_iter=5, random_state=42))

        ])

        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'vect__ngram_range': [(1, 2), (1, 3)],
            'selector__percentile': (85, 95),
            'selector__score_func': (chi2, f_classif),
            'vect__use_idf': (True, False),
        }

        skf = StratifiedShuffleSplit(y_train,3)

        grid_search = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1)
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

    def use_pipeline_with_fs(self,docs_train,y_train,docs_test,y_test):

        vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1,3), min_df=3, max_df=0.90)

        selector = SelectPercentile(score_func=chi2, percentile=85)

        combined_features = Pipeline([
                                        ("vect", vectorizer),
                                        ("feat_select", selector)
        ])

        X_features = combined_features.fit_transform(docs_train,y_train)
        X_test_features = combined_features.transform(docs_test)

        #selector.fit(X_tfidf, y_train)
        #X_selector=selector.transform(X_tfidf)

        print (X_features.shape)
        print (X_test_features.shape)

        # Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        pipeline = Pipeline([
                ('clf', SGDClassifier(loss='hinge', n_iter=5, random_state=42))

        ])

        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'clf__penalty': ('l1', 'l2')
        }

        skf = StratifiedShuffleSplit(y_train,3)

        grid_search = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1)
        clf_gs = grid_search.fit(X_features, y_train)


        # print the cross-validated scores for the each parameters set
        # explored by the grid search
        #print(clf_gs.grid_scores_)


        best_parameters, score, _ = max(clf_gs.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        print("score is %s" % score)



        y_predicted = clf_gs.predict(X_test_features)


        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        # import matplotlib.pyplot as plt
        # plt.matshow(cm)
        # plt.show()

    def plot_feature_selection(self,docs_train,y_train):


        transform = SelectPercentile(score_func=chi2)

        clf = Pipeline([('anova', transform), ('clf', SGDClassifier())])

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



if __name__ == '__main__':

    dataset = pd.read_csv('output/engrate/output_engrate_label_space_noART.csv', header=0, names=['tweets', 'class'])

    X = dataset['tweets']
    y = dataset['class']

    print (len(X))

    #stop words
    lines = open('stopwords.csv', 'r').readlines()

    my_stopwords=[]
    for line in lines:
        my_stopwords.append(line.replace("\n", ""))

    stopwords = text.ENGLISH_STOP_WORDS.union(my_stopwords)

    svm = SVM()

    """Split data using Cross Validation"""

    #docs_train,docs_test,y_train,y_test = svm.train_test_split()
    docs_train,docs_test,y_train,y_test = svm.stratified_shufflesplit()
    #docs_train,docs_test,y_train,y_test = svm.stratified_kfolds()

    """Get features"""

    X_tfidf,count_vect,tfidf_transformer = svm.get_features(docs_train)

    """SVM Classifier"""

    #y_predicted, clf = svm.train_classifier(X_tfidf,y_train,docs_test,y_test,count_vect,tfidf_transformer)

    """Use Feature Selection"""

    #y_predicted, clf = svm.use_feature_selection(X_tfidf,y_train,docs_test,y_test,count_vect,tfidf_transformer)

    """Confusion matrix"""

    #svm.confusion_matrix(y_test,y_predicted)

    """Get important features"""

    #svm.get_important_features(clf,count_vect)

    """Pipeline"""

    #svm.use_pipeline(docs_train,y_train,docs_test,y_test)
    svm.use_pipeline_with_fs(docs_train,y_train,docs_test,y_test)

    """Plot feature selection"""
    #svm.plot_feature_selection(docs_train,y_train)



