__author__ = 'yi-linghwong'

import re
import os
import sys
import itertools
from extractor import Extractor


if os.path.isfile('output/output_engrate_textonly.csv'):
  lines = open('output/output_engrate_textonly.csv','r', encoding = "ISO-8859-1").readlines()

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

    def label_tweets(self):

        if os.path.isfile('output/output_engrate_010815.csv'):
            lines = open('output/output_engrate_010815.csv', 'r').read()

        else:
            print("File not found")
            sys.exit(1)


        tweets_label = []

        print (lines)

        for line in lines:

            tweets = []
            print (line)

            spline=line.replace("\n","").split(",")

            #split string from 2nd comma until the third last comma, and then join them together into one single string
            tweets.append("".join(spline[2:len(spline)-3]))

            print (float(spline[-1]))

            if float(spline[-1]) >= 0.02:

                tweets.append('HRT')


            elif (float(spline[-1]) >= 0.04) and (float(spline[-1]) < 0.02):

                tweets.append('ART')

            else:

                tweets.append('LRT')

            print (tweets)

            tweets_label.append(tweets)

            #print (tweets_label)










ext = Extractor()
dp = DataProcessing()

'''
Clean up tweets



clean_tweets = dp.remove_sc(dp.remove_mention(dp.remove_RT(dp.remove_url(tweets))))

#print (clean_tweets)

#remove duplicates!
#clean_tweets.sort()
#ct = list(clean_tweets for clean_tweets,_ in itertools.groupby(clean_tweets))
#print (ct)

no_duplicate = []
duplicate = []

for ct in clean_tweets:
    if ct not in no_duplicate:
        no_duplicate.append(ct)
    else:
        duplicate.append(ct)


'''
#print output to csv file
#ext.printcsv_all(no_duplicate,'engrate_clean')
#ext.printcsv_all(sc_no_dup,'clean_sc')
#ext.printcsv_all(duplicate,'kepler_dup')

dp.label_tweets()


