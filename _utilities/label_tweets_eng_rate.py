__author__ = 'yi-linghwong'

################
# label tweets by engagement rate (HRT, LRT)
# subroutines: clean up tweets, calculate engagement rate, label tweets
################

import time
import sys
import os
import re
import numpy as np


class LabelTweetsEngRate():


    def clean_up_tweets(self):

        lines = open(path_to_tweet_file,'r').readlines()

        tweets = []

        for line in lines:
            spline = line.replace('\n','').split(',')
            tweets.append(spline)

        clean_tweets = []

        print ("Cleaning up tweets ...")

        for t in tweets:

            t1 = t[-1]

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
                    t6 = " ".join(words)

            # convert tweet to lower case
            t7 = t6.lower()

            t[6] = t7

            clean_tweets.append(t)

        return clean_tweets


    def get_eng_rate(self):

        tweets = self.clean_up_tweets()

        engrate_list = []

        print ("Calculating engagement rate...")

        for t in tweets:

            if len(t) == 7:

                engrate = str((np.divide(int(t[4]),int(t[2])))*100)

                engrate_list.append([t[0],t[1],t[2],t[3],t[4],t[5],engrate,t[6]])

            else:
                print ("error")
                print(t)
                pass

        f = open(path_to_store_engrate_output,'w')

        for el in engrate_list:
            f.write(','.join(el)+'\n')

        f.close()

        return engrate_list


    def label_tweets(self):

        tweets = self.get_eng_rate()

        labelled_tweets = []

        print ("Labelling tweets ...")

        print ("Length of tweet list is "+str(len(tweets)))

        for t in tweets:

            if float(t[6]) > 0.06:

                labelled_tweets.append([t[7],'HRT'])

            elif float(t[6]) < 0.0005:

                labelled_tweets.append([t[7],'LRT'])

            else:
                pass

        f = open(path_to_store_labelled_tweets, 'w')

        for lt in labelled_tweets:

            f.write(','.join(lt)+str('\n'))

        f.close()

        print ("Length of labelled tweets is "+str(len(labelled_tweets)))

        return labelled_tweets



if __name__ == "__main__":

    path_to_tweet_file = '../extracted_data/ALL_nofollcorr.csv'
    path_to_store_engrate_output = '../output/engrate/engrate_ALL_nofollcorr.csv'
    path_to_store_labelled_tweets = '../output/engrate/labelled_ALL_nofollcorr.csv'


    lt = LabelTweetsEngRate()
    #lt.clean_up_tweets()
    #lt.get_eng_rate()
    lt.label_tweets()



'''

lines = open('../followers/user_slope_space.txt','r').readlines()

slope_dict = {}

for line in lines:
    spline = line.replace('\n','').split(',')
    slope_dict[spline[0]] = spline[1]

print ("Length of slope_dict is "+str(len(slope_dict)))



###############
# create tweet list with updated follower count
###############


lines = open('../output/engrate/_old/output_engrate_MASTER.csv','r').readlines()

tweets = []
for line in lines:
    spline = line.replace('\n','').split(',')
    tweets.append(spline)

print ("Length of tweet list is "+str(len(tweets)))

updated_tweets = []

for t in tweets:

    key = t[0]

    if key in slope_dict:

        del t[6]
        updated_tweets.append(t)

print (len(updated_tweets))

f = open('../extracted_data/space_nofollcorr.csv','w')

for ut in updated_tweets:
    f.write(','.join(ut)+'\n')

f.close()

'''

