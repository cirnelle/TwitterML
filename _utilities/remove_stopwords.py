__author__ = 'yi-linghwong'

###############
# single method to remove stop words from a list of tweets
###############

import sys
import itertools
from sklearn.feature_extraction import text

lines = open('../stopwords/stopwords.csv', 'r').readlines()

my_stopwords=[]
for line in lines:
    my_stopwords.append(line.replace("\n", ""))

stopwords = text.ENGLISH_STOP_WORDS.union(my_stopwords)


lines = open('output/sydscifest/output_sydsciencefest_ALL_CLEAN.csv', 'r').readlines()

tweetlist = []
for line in lines:
    spline=line.replace("\n", "").split(",")

    #transform all words in the line to lowercase
    words = [word.lower() for word in spline[0].split()] #output ['one', 'two', 'three']

    #remove words from stopword list from the words list
    rm_stopwords = [word for word in words if word not in stopwords] #output ['two', 'three']

    #join all elements in the list (individual words) into one string
    no_stopwords = [' '.join(rm_stopwords)] #output ['two three']

    #no_stopwords.append(spline[1]) #output ['this is a tweet', 'LRT']

    #join all elements in the list (individual words) into one string
    #no_stopwords = [','.join(no_stopwords)] #output ['this is a tweet,LRT']

    tweetlist.extend(no_stopwords)#output ['this is a tweet,LRT', 'this is also a tweet,HRT']

f = open('output/sydscifest/output_sydsciencefest_ALL_CLEAN_nostop.csv', 'w')

for t in tweetlist:
    f.write(str(t)+'\n')

f.close()

