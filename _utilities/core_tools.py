__author__ = 'yi-linghwong'

##############
# a set of tools, includes:
# get average number RTs
# get engagement rate
# get histogram (eng rate vs # of tweets)
# get words (split by white space)
# get word frequency
# get lexical diversity
# get average number of words per tweet (for HRT, LRT, ART)
# get the tweets of a subset of users from a tweet dump file
# get related hashtags for a list of tweets
##############

from extractor import Extractor
import os
import sys
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import time
from sklearn.feature_extraction import text


if os.path.exists("/Users/yi-linghwong/GitHub/TwitterML/"):
    maindir = "/Users/yi-linghwong/GitHub/TwitterML/"
elif os.path.exists("/home/yiling/GitHub/GitHub/TwitterML/"):
    maindir = "/home/yiling/GitHub/GitHub/TwitterML/"
else:
    print ("ERROR --> major error")
    sys.exit(1)


#user_list=['nasa']



#ext= Extractor()
#auth = ext.loadtokens()
#api = ext.connectToAPI(auth)


class Compute():

    def get_average_RT_by_follower_number(self,users):

        ############
        # get the average number of RT, against the number of followers they have
        # i.e. get relationship between average number of RT vs follower count
        ############

        full_tweets = []
        av_list = []

        for user in users:

            full_tweets = ext.gettweets_user(user,api)
            #print (full_tweets)

            rt=[]
            followers=[]

            for t in full_tweets:
                rt.append(t[3])
                followers.append(t[2])

            av_list.append([np.average(followers),np.average(rt)])

        return (av_list)

            #print (averageRT)
            #print (averageFollowers)

            #av_list.extend()


    def get_eng_rate(self,users):

        full_tweets = []
        engrate_list = []

        for user in users:


            full_tweets = ext.gettweets_user(user,api)
            #print (full_tweets)


            for t in full_tweets:

                engrate = (np.divide(t[5],t[3]))*100

                #print text of tweet as last element to eliminate complications that comes with comma (or pipe) delimited files when processing tweets
                engrate_list.append([t[0], t[1], t[2], t[3], t[4], t[5], engrate, t[7]])

            ext.printcsv_all(engrate_list,'engrate_live')

        #return engrate_list


    def get_histogram(self):

        lines = open('updated_engrate.txt','r').readlines()

        erlist=[]
        zero_list = []

        for line in lines:
            spline=line.replace("\n", "").split(",")
            #creates a list with key and value. Split splits a string at the comma and stores the result in a list

            #some lines have unexpected line breaks which mess up the output (the last item in the list is the tweet, not the ER)


            #important to convert to float!!

            number=float(spline[6])

            if number > 0:

                erlist.append(number)

            elif number == 0:
                zero_list.append(number)

        print ("Length of list is "+str(len(erlist)))
        print ("Length of zero list is "+str(len(zero_list)))
        print (min(erlist))
        print (max(erlist))


        #plt.hist(erlist,bins=30)


        MIN, MAX = min(erlist), max(erlist)

        print (np.log10(MIN))

        plt.hist(erlist, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50))
        plt.gca().set_xscale("log")
        #need block=True to keep plot opened
        plt.xlabel("Engagement rate")
        plt.ylabel("Number of Tweets")
        plt.show(block=True)


    def get_words(self):

        words = []

        if os.path.isfile('output/misc/engrate_space_only_tweets.csv'):
            lines = open('output/misc/engrate_space_only_tweets.csv', 'r').readlines()

        else:
            print("File not found")
            sys.exit(1)

        for line in lines:
            for word in line.split():
                words.append(word.lower())

        return (words)


    def get_word_freq(self,word_list):

        word_freq = []

        lines = open('stopwords.csv', 'r').readlines()

        my_stopwords=[]
        for line in lines:
            my_stopwords.append(line.replace("\n", ""))

        stop = text.ENGLISH_STOP_WORDS.union(my_stopwords)

        #stop = stopwords.words('english')
        #stop.extend(stop_words)

        for w in word_list:
            if w not in stop:
                word_freq.append(w)

        f = open(maindir+'/output/misc/engrate_space_wordfreq.txt', 'w')

        wf = Counter(word_freq).most_common(200)

        for w in wf:
            f.write(str(w)+'\n')

        f.close()

        #print (Counter(word_freq).most_common(20))

    def lexical_diversity(self):

        lines = open('output/output_engrate_label_140815.csv', 'r', encoding = "ISO-8859-1").readlines()

        l1=[]
        l2=[]
        l3=[]


        for line in lines:
            spline=line.replace("\n","").split(",")

            words=spline[0].split()

            if (spline[1]=='LRT'):

                lexdiv=1.0*len(set(words))/len(words)

                l1.append(lexdiv)

            elif (spline[1]=='HRT'):

                lexdiv=1.0*len(set(words))/len(words)
                l2.append(lexdiv)

            else:
                lexdiv=1.0*len(set(words))/len(words)
                l3.append(lexdiv)


        average_lrt = np.average(l1)
        average_hrt = np.average(l2)
        average_art = np.average(l3)

        print (average_lrt)
        print (average_hrt)
        print (average_art)

    def average_words(self):

        lines = open('output/output_engrate_label_140815.csv', 'r', encoding = "ISO-8859-1").readlines()

        l1=[]
        l2=[]
        l3=[]


        for line in lines:
            spline=line.replace("\n","").split(",")

            words=spline[0].split()

            if (spline[1] == 'LRT'):

                l1.append(len(words))

            elif (spline[1] == 'HRT'):

                l2.append(len(words))

            else:

                l3.append(len(words))

        average_words_lrt=np.average(l1)
        average_words_hrt=np.average(l2)
        average_words_art=np.average(l3)

        print (average_words_lrt)
        print (average_words_hrt)
        print (average_words_art)


    def get_specific_user_tweets(self):

        ###Get the tweets of specified users from a tweet dump###

        lines1 = open('output/engrate/output_engrate_MASTER.csv', 'r').readlines()
        lines2 = open('user_individuals.csv', 'r').readlines()


        l1=[]

        for line in lines2:
            spline2 = line.replace('\n','').split(',')

            for line in lines1:
                spline1 = line.replace('\n','').split(',')

                if (spline1[0] == spline2[0]):
                    l1.append(line)


        f = open('output/engrate/output_engrate_individuals.csv', 'w')

        for l in l1:
            f.write(str(l))

        f.close()

    def get_related_hashtags(self):

        lines = open('output/education/output_aussieED.csv', 'r').readlines()

        hashtag_list=[]

        ##create a list of all hashtags###

        for line in lines:

            hl = [word.strip("#") for word in line.replace('\n','').split() if word.startswith("#")]
            hashtag_list.extend(hl)

        ##convert all hashtags to lowercase##
        hash_list = [ht.lower() for ht in hashtag_list]


        ##count occurences of each hashtag and list the top 10 most common##
        hash_count=Counter(hash_list).most_common(15)

        #print (hash_count)

        f=open('output/education/aussieED_related_hashtag.csv', 'w')

        for hc in hash_count:
            f.write(str(hc)+'\n')

        f.close



cp = Compute()


#word_list = cp.get_words()

#cp.get_word_freq(word_list)


#cp.get_specific_user_tweets()


#engratelist = cp.get_eng_rate(user_list)

#write file in function itself! So that if program interrupted data will be flushed to file.
#ext.printcsv_all(engratelist,'engrate_live')



cp.get_histogram()


#cp.lexical_diversity()


#cp.average_words()




#average = cp.get_average(user_list)
#ext.printcsv_all(average,'av')


#cp.get_related_hashtags()




