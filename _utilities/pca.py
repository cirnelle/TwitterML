__author__ = 'yi-linghwong'

import sys
import os
import operator
import pandas as pd
from sklearn import cross_validation
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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class PrincipalComponentAnalysis():

    def train_test_split(self):

        #################
        #Split the dataset in training and test set:
        #################

        docs_train, docs_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)

        print ("Number of data point is "+str(len(y)))

        return docs_train, docs_test, y_train, y_test


    def do_pca(self):

        # Get list of features
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1,1))
        X_CV = count_vect.fit_transform(docs_train)


        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation

        tfidf_transformer = TfidfTransformer(use_idf = True)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        # convert sparse matrix to dense data, pca only takes dense data as input!

        X_dense = X_tfidf.toarray()

        ################
        # run PCA on dense data
        ################

        pca = PCA(n_components=_n_components)
        X_new = pca.fit_transform(X_dense)

        print ("Shape of dense data is "+str(X_new.shape)) #shape = (n_samples, n_components)

        feature_names = count_vect.get_feature_names()

        print ("Number of feature is "+str(len(feature_names)))

        # the first component always has the highest variance

        components = pca.components_[0] # shape = (n_components, n_features)

        print ("Explained variance ratio: "+str(pca.explained_variance_ratio_))

        zipped = zip(components,feature_names)

        feature_pca = []

        for z in zipped:
            z = list(z)
            feature_pca.append(z)

        feature_pca.sort(reverse=True)

        feature_pca_list = []

        for fp in feature_pca:
            feature_pca_list.append([str(fp[0]),fp[1]])

        f = open(path_to_store_pca_result_file,'w')

        for fp in feature_pca_list:
            f.write(','.join(fp)+'\n')

        f.close()

        return


    def plot_variance_graph(self):

        # Get list of features
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1,1))
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation###

        tfidf_transformer = TfidfTransformer(use_idf = True)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        X_dense = X_tfidf.toarray()

        pca = PCA() # if no n_components specified, then n_components = n_features

        ###############################################################################
        # Plot the PCA spectrum

        pca.fit_transform(X_dense)
        print ("#############")
        print ("Explained variance ratio is "+str(pca.explained_variance_ratio_))

        #plt.figure(1, figsize=(4, 3))
        plt.clf()
        #plt.axes([.2, .2, .7, .7])
        plt.plot(pca.explained_variance_, linewidth=2)
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('explained_variance_')
        plt.show()

        return


################
# variables
################

path_to_labelled_file = '../output/features/space/labelled_combined.csv'
path_to_stopword_file = '../stopwords/stopwords.csv'
path_to_store_pca_result_file = '../output/pca/space_combined.csv'

# for pca
_n_components = 5


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

    pc = PrincipalComponentAnalysis()

    ###################
    # split data into training and test set
    ###################

    docs_train,docs_test,y_train,y_test = pc.train_test_split()

    ###################
    # do PCA and save results to file
    ###################

    pc.do_pca()

    ###################
    # plot n_components vs variance graph
    ###################

    #pc.plot_variance_graph()

