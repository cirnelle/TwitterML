__author__ = 'yi-linghwong'

import sys
import pandas as pd
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
from sklearn import metrics
import numpy as np
import scipy as sp
from sklearn.feature_extraction import text


if __name__ == "__main__":

    dataset = pd.read_csv('output/output_engrate_label_080815_noART.csv', header=0, names=['tweets', 'class'])

    X = dataset['tweets']
    y = dataset['class']

    print (len(X))

    #stop words
    lines = open('stopwords.csv', 'r').readlines()

    my_stopwords=[]
    for line in lines:
        my_stopwords.append(line.replace("\n", ""))

    stopwords = text.ENGLISH_STOP_WORDS.union(my_stopwords)

    ####Split the dataset in training and test set:###
    #docs_train, docs_test, y_train, y_test = train_test_split(
        #X, y, test_size=0.20, random_state=42)


    ###Stratified ShuffleSplit cross validation iterator###
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





    ###Stratified K-Folds cross validation iterator###
    #Provides train/test indices to split data in train test sets.
    #This cross-validation object is a variation of KFold that returns stratified folds.
    # The folds are made by preserving the percentage of samples for each class.

    '''
    skf = StratifiedKFold(y, n_folds=5)

    print (len(skf))
    print(skf)

    for train_index, test_index in skf:
        print("TRAIN:", train_index, "TEST:", test_index)
        docs_train, docs_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    '''

    ### Get list of features
    count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1,3))
    X_CV = count_vect.fit_transform(docs_train)
    ### print number of unique words (n_features)
    print (X_CV.shape)

    ### Get list of features
    #sys.stdout = open('output/output_features.txt', "w")
    #print (count_vect.get_feature_names())
    #print (count_vect.vocabulary_)
    #print (len(count_vect.vocabulary_))
    f1 = open('output/vocab.txt', 'w')
    temp=[]
    for key in count_vect.vocabulary_:
        temp.append(key)


    temp = sorted(temp)

    for t in temp:
        f1.write(t+'\n')

    f1.close()


    ###tfidf transformation
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_CV)
    #print(X_tfidf.shape)

    ###training the classifier
    clf = ExtraTreesClassifier().fit(X_tfidf, y_train)
    feat_imp = clf.feature_importances_
    f = open('output/extraTree_feat_imp_all.txt', 'w')
    for fea in feat_imp:

        f.write(str(fea)+'\n')
    f.close()

    lines = open('output/extraTree_feat_imp_all.txt', 'r').readlines()
    lines2 = open('output/vocab.txt', 'r').readlines()

    sortli = sorted(range(len(feat_imp)), key=lambda i:feat_imp[i], reverse=True)[:100]


    for i in sortli:
        f = open('output/extraTree_feature_importance.csv', 'a')
        f.write(lines[i].replace('\n','')+'\t'+lines2[i].replace('\n','')+'\n')
    f.close()


    ##test clf on test data

    X_test_CV = count_vect.transform(docs_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_CV)

    scores = cross_val_score(clf, X_test_tfidf, y_test, cv=5, scoring='f1_weighted')
    print ("Cross validation score:%s " % scores)

    y_predicted = clf.predict(X_test_tfidf)

    #print the mean accuracy on the given test data and labels.
    print ("Classifier score is: %s " % clf.score(X_test_tfidf,y_test))

    # Print and plot the confusion matrix

    print(metrics.classification_report(y_test, y_predicted))
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)



'''

    # Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
    pipeline = Pipeline([
            ('vect', TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=0.90)),
            ('clf', ExtraTreesClassifier()),
    ])


    # Build a grid search to find the best parameter
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
        'vect__ngram_range': [(1, 2), (1, 3)],
        'vect__use_idf': (True, False),
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

    '''





