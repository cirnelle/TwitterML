__author__ = 'yi-linghwong'

import collections
import nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import sys

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')


negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

#print (negfeats) #output [(word_dict,'LRT'), ...]


negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

#f=open('temp.txt', 'w')

reflist = []
testlist = []

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    reflist.append(label)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
    testlist.append(observed)

#f.write(str(reflist))
#f.close()


print ('pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))
print ('pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos']))
print ('pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos']))
print ('neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg']))
print ('neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))
print ('neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg']))

cm = nltk.ConfusionMatrix(reflist, testlist) #(y_test, y_predicted)
print(cm.pretty_format(sort_by_count=True, show_percents=False, truncate=2))