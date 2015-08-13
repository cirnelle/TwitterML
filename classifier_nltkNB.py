__author__ = 'yi-linghwong'

import nltk.corpus
from nltk.classify import NaiveBayesClassifier
import random
from nltk.corpus import movie_reviews
import sys
from collections import Counter
import itertools
from nltk.util import ngrams

#NLTK Naive Bayes classifier expects a list of tuples as input##
##Creates list of tuples in the format [(['one','two','one two', ...], 'HRT'), ...]


lines = open('output/output_engrate_label_080815_noART_noStop.csv', 'r').readlines()

l3=[] #list of tuples for HRT
l4=[] #list of tuples for LRT
l5=[] #list of all words for HRT
l6=[] #list of all words for LRT

for line in lines:
    spline=line.replace("\n", "").split(",")
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    #create a list of word which includes ngrams
    n=4
    l1=[]
    l2=[]

    if (spline[1] == 'HRT'):
        for i in range(1,n):
            n_grams = ngrams(spline[0].split(), i) #output [('one', 'two'), ('two', 'three'), ('three', 'four')]
            #join the elements within the list together
            gramify = [' '.join(x) for x in n_grams] #output ['one two', 'two three', 'three four']
            #l1 contains list of ngram words for only that specific line!
            l1.extend(gramify)
            #l5 contains list of all words for HRT
            l5.extend(gramify)
            i=i+1


        l3.append(l1)

    elif spline[1] == 'LRT':
        for i in range(1,n):
            n_grams = ngrams(spline[0].split(), i)
            gramify = [' '.join(x) for x in n_grams]
            #l2 contains list of ngram words for only that specific line!
            l2.extend(gramify)
            #l6 contains list of all words for LRT
            l6.extend(gramify)
            i=i+1

        l4.append(l2)


HRT = [(tweets,'HRT')
        for tweets in l3]

LRT = [(tweets, 'LRT')
       for tweets in l4]

print ("Creating word list...")

words_list = l5 + l6


documents = HRT + LRT
random.shuffle(documents)


#all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

all_words = nltk.FreqDist(w.lower() for w in words_list)

word_features = list(all_words)[:5000]

def document_features(wordlist):
    document_words = set(wordlist)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

print ("Creating featuresets...")
featuresets = [(document_features(d), c) for (d,c) in documents]

n = int(0.2*len(documents))

train_set, test_set = featuresets[n:], featuresets[:n]

print ("Putting classifier to work...")
classifier = nltk.NaiveBayesClassifier.train(train_set)

print (nltk.classify.accuracy(classifier, test_set))



