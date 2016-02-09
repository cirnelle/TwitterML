__author__ = 'yi-linghwong'

###############
# methods for tweet preprocessing, includes:
# remove URL, RT, mention, special characters, stopwords, remove all of the above ('cleanup') plus single characters
# label tweets ('LRT', 'HRT', 'ART')
###############

import re
import os
import sys
import itertools
from extractor import Extractor
#from nltk.corpus import stopwords
from sklearn.feature_extraction import text

lines = open('../stopwords/stopwords.csv', 'r').readlines()

my_stopwords=[]
for line in lines:
    my_stopwords.append(line.replace("\n", ""))

stop_words = text.ENGLISH_STOP_WORDS.union(my_stopwords)


if os.path.isfile('output/sydscifest/sydsciencefest_only_tweets.csv'):
  lines = open('output/sydscifest/sydsciencefest_only_tweets.csv','r', encoding = "ISO-8859-1").readlines()

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

#stop_words = ['ve', 're', 'll', 'amp']

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
            clean_tweets.append(text)


        return clean_tweets

    def remove_stopwords(self,tweets):

        clean_tweets=[]


        for t in tweets:
            no_stop=[] #important

            for w in t.split():
                #remove single characters and stop words
                if (len(w.lower())>=2) and (w.lower() not in stop_words):
                    no_stop.append(w.lower())


                    #join the list of words together into a string
                    text = " ".join(no_stop)



            clean_tweets.append([text])

        return clean_tweets


    def cleanup(self):

        if os.path.isfile('output/sydscifest/sydsciencefest_only_tweets.csv'):
            lines = open('output/sydscifest/sydsciencefest_only_tweets.csv', 'r', encoding = "ISO-8859-1").readlines()

        else:
            print("File not found")
            sys.exit(1)


        tweets_label = []

        tweets = []
        for line in lines:

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
            t5 = re.sub("[^A-Za-z0-9]+",' ', t4)


            #remove single characters

            words=[]
            for word in t5.split():


                if (len(word)>=2):
                    words.append(word)



                    #join the list of words together into a string
                _ = " ".join(words)

                #print (_)


            tweets.append([_])


        ext.printcsv_all(tweets,'sydsciencefest_ALL_CLEAN')


    def label_tweets(self):

        if os.path.isfile('output/engrate/output_engrate_MASTER.csv'):
            lines = open('output/engrate/output_engrate_MASTER.csv', 'r').readlines()

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
            t5 = re.sub("[^A-Za-z0-9]+",' ', t4)

            #remove single characters
            words=[]
            for word in t5.split():
                if (len(word)>=2):
                    words.append(word)

                    #join the list of words together into a string
                    _ = " ".join(words)

            tweets.append(_)

            if (len(spline)>=8):


                if float(spline[6]) >= 0.055:

                    tweets.append('HRT')


                elif (float(spline[6]) >= 0.0012) and (float(spline[6]) < 0.0):

                    tweets.append('ART')

                else:

                    tweets.append('LRT')

                tweets_label.append(tweets)

            else:
                print ("skipping")





        ext.printcsv_all(tweets_label,'engrate_label_MASTER')

        #return (tweets_label)


if __name__ == "__main__"

    ext = Extractor()
    dp = DataProcessing()

    #dp.cleanup()

    dp.label_tweets()


    """Clean up tweets"""

    '''

    clean_tweets = dp.remove_mention(dp.remove_RT(dp.remove_url(tweets)))

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
    ext.printcsv_all(no_duplicate,'sydsciencefest_ALL_CLEAN')
    #ext.printcsv_all(sc_no_dup,'clean_sc')
    #ext.printcsv_all(duplicate,'kepler_dup')

    '''



