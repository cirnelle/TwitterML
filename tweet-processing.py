__author__ = 'yi-linghwong'

import re
import os
import sys
import itertools
from extractor import Extractor


if os.path.isfile('output_kepler.csv'):
  lines = open('output_kepler.csv','r').readlines()

else:
    print ("File not found")
    sys.exit(1)

tweets = []

for line in lines:
    spline=line.replace("\n", "").split("\n")
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    #remove empty lines!
    if line.rstrip():
        tweets.extend(spline)


class DataProcessing():

    def remove_url(self,tweets):

        clean_tweets = []

        for t in tweets:

            # r"(?:\@|https?\://)\S+" removes both URL and mentions at the same time
            text = re.sub(r'(?:https?\://)\S+', '', t)
            clean_tweets.append(text)


        return clean_tweets

    def remove_RT(self,tweets):

        clean_tweets = []

        for t in tweets:
            text = t.replace('RT','')
            clean_tweets.append(text)


        return clean_tweets


    def remove_mention(self,tweets):

        clean_tweets = []

        for t in tweets:

            text = re.sub(r'(?:\@)\S+', '', t)
            #creates a list with square brackets! So printcsv (which takes list of a list as input) could process this data.
            clean_tweets.append(text)


        return clean_tweets

    def remove_sc(self,tweets):

        clean_tweets = []

        for t in tweets:

            text = re.sub('[^A-Za-z0-9]+',' ', t)
            #creates a list with square brackets! So printcsv (which takes list of a list as input) could process this data.
            clean_tweets.append([text])


        return clean_tweets



ext = Extractor()
dp = DataProcessing()
clean_tweets = dp.remove_sc(dp.remove_mention(dp.remove_RT(dp.remove_url(tweets))))
print (clean_tweets)

#remove duplicates!
clean_tweets.sort()
ct = list(clean_tweets for clean_tweets,_ in itertools.groupby(clean_tweets))
print (ct)

#print output to csv file
ext.printcsv_all(ct,'clean')




