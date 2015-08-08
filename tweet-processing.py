__author__ = 'yi-linghwong'

import re
import os
import sys
import itertools
from extractor import Extractor
from nltk.corpus import stopwords


if os.path.isfile('output/output_kepler_after.csv'):
  lines = open('output/output_kepler_after.csv','r', encoding = "ISO-8859-1").readlines()

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

stop_words = ['s', 'd']

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

        if os.path.isfile('output/output_engrate_070815.csv'):
            lines = open('output/output_engrate_070815.csv', 'r').readlines()

        else:
            print("File not found")
            sys.exit(1)


        tweets_label = []

        for line in lines:

            tweets = []

            spline=line.replace("\n","").split(",")




            t1 = spline[-1]

            #split string from 2nd comma until the third last comma, and then join them together into one single string
            #t1 = "".join(spline[2:len(spline)-3])

            #remove URLs
            t2 = re.sub(r'(?:https?\://)\S+', '', t1)
            #remove 'RT'
            t3 = t2.replace('RT','')
            #remove mentions
            t4 = re.sub(r'(?:\@)\S+', '', t3)
            #remove special characters
            t5 = re.sub("[^A-Za-z]+",' ', t4)

            #remove single characters
            words=[]
            for word in t5.split():
                if (len(word)>=2):
                    words.append(word)

                    #join the list of words together into a string
                    _ = " ".join(words)

            tweets.append(_)

            if (len(spline)>=8):


                if float(spline[6]) >= 0.025:

                    tweets.append('HRT')


                elif (float(spline[6]) >= 0.0012) and (float(spline[6]) < 0.025):

                    tweets.append('ART')

                else:

                    tweets.append('LRT')

                tweets_label.append(tweets)

            else:
                print ("skipping")





        ext.printcsv_all(tweets_label,'engrate_label_tmp')

        #return (tweets_label)



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



#print output to csv file
ext.printcsv_all(no_duplicate,'kepler_after_clean')
#ext.printcsv_all(sc_no_dup,'clean_sc')
#ext.printcsv_all(duplicate,'kepler_dup')

'''

dp.label_tweets()


