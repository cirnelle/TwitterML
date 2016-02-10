__author__ = 'yi-linghwong'

#################
# label tweets by n most and least retweeted tweets per user
# subroutines: clean up tweet
#################

import sys
import os
import re

class BinTweets():

    def create_top_bottom_tweets(self):

        lines = open(path_to_user_list,'r').readlines()

        users = []

        for line in lines:
            spline = line.replace('\n','').split(',')
            users.append(spline[0])

        print ("Length of user list is "+str(len(users)))

        tweets = []

        lines = open(path_to_tweet_file,'r').readlines()

        for line in lines:
            spline = line.replace('\n','').split(',')
            tweets.append(spline)

        print ("Length of tweet list is "+str(len(tweets)))

        top_bottom_tweets = []
        labelled_tweets = []

        for u in users:

            user_tweet = []

            for t in tweets:

                if t[0] == u:

                    user_tweet.append(t)

                else:
                    continue

                # sort tweets by number of retweets in descending order

                user_tweet = sorted(user_tweet, key=lambda t: int(t[4]), reverse=True)

            # how many percent we want to cut

            cut = round(0.09 * len(user_tweet))

            # calculate length of half of the tweets per user
            # to label half HRT, and the other half LRT

            half = round(0.5*len(top_bottom_tweets))

            for ut in user_tweet[:cut]:
                ut.append('HRT')
                labelled_tweets.append(ut)

            for ut in user_tweet[-cut:]:
                ut.append('LRT')
                labelled_tweets.append(ut)

        print ("Length of labelled tweets is "+str(len(labelled_tweets)))

        f = open('../output/engrate/topRT_space.csv','w')

        for lt in labelled_tweets:

            f.write(','.join(lt)+'\n')

        f.close()

        return labelled_tweets

    def clean_up(self):

        tweets = self.create_top_bottom_tweets()

        clean_tweets = []

        for t in tweets:

            t1 = t[-2]

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

            t7 = t6.lower()

            clean_tweets.append([t7,t[-1]])

        print ("Length of clean tweets is "+str(len(clean_tweets)))

        f = open(path_to_store_labelled_tweets,'w')

        for ct in clean_tweets:
            f.write(','.join(ct)+'\n')

        f.close()

if __name__ == "__main__":

    path_to_user_list = '../followers/user_slope.txt'
    path_to_tweet_file = '../extracted_data/ALL_nofollcorr.csv'
    path_to_store_labelled_tweets = '../output/engrate/labelled_ALL_topRT.csv'

    bt = BinTweets()
    #bt.create_top_bottom_tweets()
    bt.clean_up()