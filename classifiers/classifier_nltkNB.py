__author__ = 'yi-linghwong'

import nltk.corpus
from nltk.classify import NaiveBayesClassifier
import collections
import nltk.metrics
import random
from nltk.corpus import movie_reviews
import sys
from collections import Counter
import itertools
from nltk.util import ngrams

#NLTK Naive Bayes classifier expects a list of tuples as input##
##Creates list of tuples in the format [(['one','two','one two', ...], 'HRT'), ...]


lines = open('output/engrate/output_engrate_label_080815_noART_noStop.csv', 'r').readlines()

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


#Create list of tuples for HRT, i.e [(['word', 'word2'], 'HRT')]
HRT = [(tweets,'HRT')
        for tweets in l3]

#Create list of tuples for LRT, i.e [(['word', 'word2'], 'LRT')]
LRT = [(tweets, 'LRT')
       for tweets in l4]

print ("Creating word list...")

words_list = l5 + l6 #list of all words, i.e. ['word, 'two words', 'one two three', ...]
print (len(words_list))


documents = HRT + LRT #complete list of tuples for all tweets
random.shuffle(documents)


all_words = nltk.FreqDist(w.lower() for w in words_list)

word_features = list(all_words)[:500]

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
print (classifier.show_most_informative_features())


"""Confusion matrix, precision and recall"""


print ('train on %d instances, test on %d instances' % (len(train_set), len(test_set)))

classifier = nltk.NaiveBayesClassifier.train(train_set)
refsets = collections.defaultdict(set) #refset is a dictionary of set, {'HRT':set(1,3,4,6,...)} where 'HRT' is key, and (1,3,4,6) is value
testsets = collections.defaultdict(set)

reflist = [] #a list of known labels
testlist = [] #a list of predicted labels

for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    reflist.append(label)
    observed = classifier.classify(feats) #returns the predicted labels for each tweet (equivalent to y_predicted)
    testsets[observed].add(i)
    testlist.append(observed)

print (len(reflist), len(testlist))

print ('HRT precision:', nltk.metrics.precision(refsets['HRT'], testsets['HRT']))
print ('HRT recall:', nltk.metrics.recall(refsets['HRT'], testsets['HRT']))
print ('HRT F-measure:', nltk.metrics.f_measure(refsets['HRT'], testsets['HRT']))
print ('LRT precision:', nltk.metrics.precision(refsets['LRT'], testsets['LRT']))
print ('LRT recall:', nltk.metrics.recall(refsets['LRT'], testsets['LRT']))
print ('LRT F-measure:', nltk.metrics.f_measure(refsets['LRT'], testsets['LRT']))

cm = nltk.ConfusionMatrix(reflist,testlist) #(y_test, y_predicted)
print(cm.pretty_format(sort_by_count=True, show_percents=False, truncate=2))



