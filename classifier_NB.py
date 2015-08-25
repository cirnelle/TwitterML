__author__ = 'yi-linghwong'

import sys
import operator
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
#from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
#from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn import metrics
import numpy as np
import scipy as sp
from sklearn.feature_extraction import text


class NaiveBayes():


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

        ###training the classifier###
        clf = MultinomialNB(alpha=0.5).fit(X_tfidf, y_train)

        print ("Fitting data ...")
        clf.fit(X_tfidf, y_train)

        """test clf on test data"""


        X_test_CV = count_vect.transform(docs_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        scores = cross_val_score(clf, X_test_tfidf, y_test, cv=3, scoring='f1_weighted')
        print ("Cross validation score:%s " % scores)

        y_predicted = clf.predict(X_test_tfidf)

        """print the mean accuracy on the given test data and labels"""

        print ("Classifier score is: %s " % clf.score(X_test_tfidf,y_test))

        return y_predicted, clf


    def use_feature_selection(self,X_tfidf,y_train,docs_test,y_test,count_vect,tfidf_transformer):

        # Create the RFE object and compute a cross-validated score.
        nb = MultinomialNB(alpha=0.5)

        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        #rfecv = RFE(estimator=nb, n_features_to_select=10000, step=1) #use recursive feature elimination
        rfecv = RFECV(estimator=nb, step=1, cv=3, scoring='accuracy')#use recursive feature elimination with cross validation

        print ("Fitting data with RFECV...")
        rfecv.fit(X_tfidf, y_train)
        X_rfecv=rfecv.transform(X_tfidf)
        print (X_rfecv.shape)

        print("Optimal number of features : %d" % rfecv.n_features_)

        clf = MultinomialNB(alpha=0.5).fit(X_rfecv, y_train)

        """test clf on test data"""


        X_test_CV = count_vect.transform(docs_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_CV)
        X_test_rfecv = rfecv.transform(X_test_tfidf)

        y_predicted = clf.predict(X_test_rfecv)

        """print the mean accuracy on the given test data and labels"""

        print ("Classifier score is: %s " % clf.score(X_test_rfecv,y_test))

        scores = cross_val_score(clf, X_test_tfidf, y_test, cv=3, scoring='f1_weighted')
        print ("Cross validation score:%s " % scores)

        return y_predicted

    def cv_and_train(self):


        """Stratified ShuffleSplit cross validation iterator"""
        #Provides train/test indices to split data in train test sets.
        #This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds.
        #The folds are made by preserving the percentage of samples for each class.


        sss = StratifiedShuffleSplit(y, 5, test_size=0.2, random_state=42)

        print (len(sss))
        print (sss)

        for train_index, test_index in sss:
            #print("TRAIN:", train_index, "TEST:", test_index)
            docs_train, docs_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1,3))
            X_CV = count_vect.fit_transform(docs_train)
            ### print number of unique words (n_features)
            print (X_CV.shape)
            tfidf_transformer = TfidfTransformer()
            X_tfidf = tfidf_transformer.fit_transform(X_CV)

            clf = MultinomialNB(alpha=0.5).fit(X_tfidf, y_train)

            print ("Fitting data ...")
            clf.fit(X_tfidf, y_train)

            X_test_CV = count_vect.transform(docs_test)
            X_test_tfidf = tfidf_transformer.transform(X_test_CV)



            y_predicted = clf.predict(X_test_tfidf)

            scores = cross_val_score(clf, X_test_tfidf, y_test, cv=3, scoring='f1_weighted')
            print ("Cross validation score:%s " % scores)

            print ("Classifier score is: %s " % clf.score(X_test_tfidf,y_test))
            print(metrics.classification_report(y_test, y_predicted))
            cm = metrics.confusion_matrix(y_test, y_predicted)
            print(cm)


    def confusion_matrix(self,y_test,y_predicted):

        """Print and plot the confusion matrix"""

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)


    def get_important_features(self,clf,count_vect):

        """Prints features with the highest coefficient values, per class"""

        n=10

        class_labels = clf.classes_
        fb_hrt=clf.feature_log_prob_[0] ##feature probability for HRT
        fb_lrt=clf.feature_log_prob_[1] ##feature probability for LRT
        feature_names = count_vect.get_feature_names()

        """the next two lines are for printing the highest feat_probability for each class"""
        topn_class1 = sorted(zip(fb_hrt, feature_names))[-n:]
        topn_class2 = sorted(zip(fb_lrt, feature_names))[-n:]


        """Most important features are the ones where the difference between feat_prob are the biggest"""

        diff = [abs(a-b) for a,b in zip(fb_hrt,fb_lrt)]

        ##sort the list by the value of the difference, and return index of that element###
        sortli = sorted(range(len(diff)), key=lambda i:diff[i], reverse=True)[:100]


        ##print out the feature names and their corresponding classes



        imp_feat=[]
        for i in sortli:

            if fb_hrt[i]>fb_lrt[i]:
                imp_feat.append('HRT '+str(feature_names[i]))
            else:
                imp_feat.append('LRT '+str(feature_names[i]))

        imp_feat=sorted(imp_feat)

        f4=open('output/feature_importance/nb_important_feat.csv', 'w')

        for imf in imp_feat:
            f4.write(imf+'\n')

        f4.close()


        """HELPER FILES"""

        f=open('output/feature_importance/nb_feature_names.txt', 'w')

        for fn in feature_names:
            f.write(str(fn)+'\n')
        f.close()

        f1=open('output/feature_importance/nb_coef.txt', 'w')

        for c in clf.coef_[0]:
            f1.write(str(c)+'\n')

        f1.close()

        f2=open('output/feature_importance/nb_feature_prob_0.csv', 'w')
        for fb in clf.feature_log_prob_[0]:
            f2.write(str(fb)+'\n')
        f2.close()

        f3=open('output/feature_importance/nb_feature_prob_1.csv', 'w')
        for fb in clf.feature_log_prob_[1]:
            f3.write(str(fb)+'\n')
        f3.close()


    def temp(self):

        """TO REMOVE: print the highest feat_probability for each class"""


        for (coef, feat) in topn_class1:
            print (class_labels[0], coef, feat)

        for coef, feat in (topn_class2):
            print (class_labels[1], coef, feat)



        """If there are more than two classes"""


        def most_informative_feature_for_class(classlabel, n=10):
            labelid = list(clf.classes_).index(classlabel)
            feature_names = count_vect.get_feature_names()
            topn = sorted(zip(clf.coef_[labelid], feature_names))[-n:]

            for coef, feat in topn:
                print (classlabel, feat, coef)

        print (most_informative_feature_for_class('HRT'))

        print (most_informative_feature_for_class('LRT'))



    def use_pipeline(self,docs_train,y_train,docs_test,y_test):

        """Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent"""


        pipeline = Pipeline([
                ('vect', TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=0.90)),
                ("selector", SelectPercentile(score_func=chi2)),
                ('clf', MultinomialNB()),
        ])


        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'vect__ngram_range': [(1, 2), (1, 3)],
            'vect__use_idf': (True, False),
            'selector__score_func': (chi2, f_classif),
            'selector__percentile': (85, 95),
            'clf__alpha': (0.4, 0.5)
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


if __name__ == '__main__':

    dataset = pd.read_csv('output/engrate/output_engrate_label_space_noART.csv', header=0, names=['tweets', 'class'])

    X = dataset['tweets']
    y = dataset['class']

    print (len(X))

    #stopwords

    lines = open('stopwords.csv', 'r').readlines()

    my_stopwords=[]
    for line in lines:
        my_stopwords.append(line.replace("\n", ""))

    stopwords = text.ENGLISH_STOP_WORDS.union(my_stopwords)

    nb = NaiveBayes()

    """Split data using Cross Validation"""

    #docs_train,docs_test,y_train,y_test = nb.train_test_split()
    docs_train,docs_test,y_train,y_test = nb.stratified_shufflesplit()
    #docs_train,docs_test,y_train,y_test = nb.stratified_kfolds()


    """Get features"""

    X_tfidf,count_vect,tfidf_transformer = nb.get_features(docs_train)

    """CV and train"""

    #nb.cv_and_train()


    """NB Classifier"""
    #y_predicted, clf = nb.train_classifier(X_tfidf,y_train,docs_test,y_test,count_vect,tfidf_transformer)

    """Use Feature Selection"""
    #y_predicted = nb.use_feature_selection(X_tfidf,y_train,docs_test,y_test,count_vect,tfidf_transformer)

    """Confusion matrix"""

    #nb.confusion_matrix(y_test,y_predicted)

    """Feature importance"""

    #nb.get_important_features(clf,count_vect)

    """Pipeline"""

    nb.use_pipeline(docs_train,y_train,docs_test,y_test)








