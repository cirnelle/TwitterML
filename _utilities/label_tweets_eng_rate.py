__author__ = 'yi-linghwong'

################
# label tweets by engagement rate (HER, LER)
# subroutines: calculate engagement rate, label tweets
################

import time
import sys
import os
import re
import numpy as np


class LabelTweetsEngRate():

    def get_eng_rate(self):

        lines = open(path_to_preprocessed_tweet_file,'r').readlines()

        tweets = []

        for line in lines:
            spline = line.replace('\n','').split(',')
            tweets.append(spline)

        print ()
        print ("Length of tweet list is "+str(len(tweets)))

        for line in lines[:1]:
            spline = line.replace('\n','').split(',')
            length = len(spline)

        print ("Number of element per line is "+str(length))

        engrate_list = []

        print ("Calculating engagement rate...")

        for t in tweets[1:]:

            if len(t) == length:

                engagement = int(t[5])+(0.5*int((t[6])))

                engrate = str((np.divide(engagement,int(t[3])))*100)

                engrate_list.append([t[0],t[1],t[2],t[3],t[4],t[5],t[6],engrate,t[7],t[8]])

            else:
                print ("error")
                print(t)
                pass

        f = open(path_to_store_engrate_output,'w')

        for el in engrate_list:
            f.write(','.join(el)+'\n')

        f.close()

        return engrate_list


    def get_eng_rate_raw_tweets(self):

        lines = open(path_to_raw_tweet_file,'r').readlines()

        tweets = []

        for line in lines:
            spline = line.replace('\n','').split(',')
            tweets.append(spline)

        print ()
        print ("Length of tweet list is "+str(len(tweets)))

        for line in lines[:1]:
            spline = line.replace('\n','').split(',')
            length = len(spline)

        print ("Number of element per line is "+str(length))

        engrate_list = []

        print ("Calculating engagement rate...")

        for t in tweets[1:]:

            if len(t) == length:

                engagement = int(t[5])+(0.5*int((t[6])))

                engrate = str((np.divide(engagement,int(t[3])))*100)

                engrate_list.append([t[0],t[1],t[2],t[3],t[4],t[5],t[6],engrate,t[7],t[8]])

            else:
                print ("error")
                print(t)
                pass

        f = open(path_to_store_engrate_output_raw,'w')

        for el in engrate_list:
            f.write(','.join(el)+'\n')

        f.close()

        return engrate_list


    def label_tweets(self):

        tweets = self.get_eng_rate()

        labelled_tweets = []
        high_er = []
        low_er = []

        print ("#############################")
        print ("Labelling preprocessed tweets ...")

        for t in tweets:

            if float(t[7]) > her_boundary:

                labelled_tweets.append([t[9],'HER'])
                high_er.append([t[9],'HER'])

            elif float(t[7]) < ler_boundary:

                labelled_tweets.append([t[9],'LER'])
                low_er.append([t[9],'LER'])

            else:
                pass


        print ("Length of high ER list is "+str(len(high_er)))
        print ("Length of low ER list is "+str(len(low_er)))


        f = open(path_to_store_labelled_tweets, 'w')

        for lt in labelled_tweets:

            f.write(','.join(lt)+str('\n'))

        f.close()

        print ("Length of labelled tweets is "+str(len(labelled_tweets)))

        return labelled_tweets


    def label_tweets_raw(self):

        tweets = self.get_eng_rate_raw_tweets()

        labelled_tweets = []
        high_er = []
        low_er = []

        print ("#############################")
        print ("Labelling raw tweets ...")

        for t in tweets:

            if float(t[7]) > her_boundary:

                labelled_tweets.append([t[9],'HER',t[8]])
                high_er.append([t[9],'HER'])

            elif float(t[7]) < ler_boundary:

                labelled_tweets.append([t[9],'LER',t[8]])
                low_er.append([t[9],'LER'])

            else:
                pass

        print ("Length of high ER list is "+str(len(high_er)))
        print ("Length of low ER list is "+str(len(low_er)))

        f = open(path_to_store_labelled_tweets_raw, 'w')

        for lt in labelled_tweets:

            f.write(','.join(lt)+'\n')

        f.close()

        print ("Length of labelled posts is "+str(len(labelled_tweets)))

        return labelled_tweets


################
# variables
################

path_to_preprocessed_tweet_file = '../tweets/others/preprocessed_politics.csv'
path_to_store_engrate_output = '../output/engrate/others/engrate_politics.csv'
path_to_store_labelled_tweets = '../output/engrate/others/labelled_politics.csv'

# for LIWC
path_to_raw_tweet_file = '../tweets/others/raw_politics.csv'
path_to_store_engrate_output_raw = '../output/engrate/others/engrate_politics_raw.csv'
path_to_store_labelled_tweets_raw = '../output/engrate/others/labelled_politics_raw.csv'

# engrate parameters
her_boundary = 0.098
ler_boundary = 0.0016


if __name__ == "__main__":


    lt = LabelTweetsEngRate()

    #lt.get_eng_rate()
    lt.label_tweets()

    #lt.get_eng_rate_raw_tweets()
    lt.label_tweets_raw()





