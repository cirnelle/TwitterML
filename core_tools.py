__author__ = 'yi-linghwong'

from extractor import Extractor
import os
import sys
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import time


if os.path.exists("/Users/yi-linghwong/GitHub/TwitterML/"):
    maindir = "/Users/yi-linghwong/GitHub/TwitterML/"
elif os.path.exists("/home/yiling/GitHub/GitHub/TwitterML/"):
    maindir = "/home/yiling/GitHub/GitHub/TwitterML/"
else:
    print ("ERROR --> major error")
    sys.exit(1)

if os.path.isfile('users_temp.txt'):
  lines = open('users_temp.txt','r').readlines()

else:
    print ("File not found")
    sys.exit(1)

user_list=[]

for line in lines:
    spline=line.replace("\n", "").split(",")
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    user_list.append(spline[0])

stop_words = ['2', '0', '1', 'b', 'amp', 'one', 'today', 'via', 'us', 'sw', 'get', 'dr', 'ks', 'sowetoshutdown']

#user_list=['nasa']



ext= Extractor()
auth = ext.loadtokens()
api = ext.connectToAPI(auth)


class Compute():

    def get_average(self,users):

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
                engrate_list.append([t[0], t[1], t[3], t[4], t[5], t[6], engrate, t[2]])

            ext.printcsv_all(engrate_list,'engrate_live')

        #return engrate_list


    def get_histogram(self,engrate_list):

        #list with only engagement rate
        #er_list = []

        #for er in engrate_list:

            #er_list.append(er[2])

        #print (er_list)

        plt.hist(engrate_list,bins=30)
        #need block=True to keep plot opened
        plt.xlabel("Engagement rate")
        plt.ylabel("Number of Tweets")
        plt.show(block=True)


    def get_words(self):



        words = []

        if os.path.isfile('output/sydscifest/output_sydsciencefest_ALL_CLEAN.csv'):
            lines = open('output/sydscifest/output_sydsciencefest_ALL_CLEAN.csv', 'r').readlines()

        else:
            print("File not found")
            sys.exit(1)

        for line in lines:
            for word in line.split():
                words.append(word.lower())

        return (words)


    def get_word_freq(self,word_list):

        word_freq = []
        stop = stopwords.words('english')
        stop.extend(stop_words)

        for w in word_list:
            if w not in stop:
                word_freq.append(w)

        f = open(maindir+'/output/sydscifest/sydscifest_wordfreq.txt', 'w')

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
        lines1 = open('output/engrate/output_engrate_MASTER.csv', 'r').readlines()
        lines2 = open('user_space.csv', 'r').readlines()


        l1=[]

        for line in lines2:
            spline2 = line.replace('\n','').split(',')

            for line in lines1:
                spline1 = line.replace('\n','').split(',')

                if (spline1[0] == spline2[0]):
                    l1.append(line)


        f = open('output/engrate/output_engrate_space.csv', 'w')

        for l in l1:
            f.write(str(l))

        f.close()


cp = Compute()

'''
Get word frequency
'''
word_list = cp.get_words()

cp.get_word_freq(word_list)

'''
Get specific user tweet
'''

#cp.get_specific_user_tweets()


'''
Get engagement rate
'''


#engratelist = cp.get_eng_rate(user_list)

#write file in function itself! So that if program interrupted data will be flushed to file.
#ext.printcsv_all(engratelist,'engrate_live')



"""Get histogram"""

'''

if os.path.isfile('output/output_engrate_individuals.csv'):
  lines = open('output/output_engrate_individuals.csv','r').readlines()

else:
    print ("File not found")
    sys.exit(1)


erlist=[]

for line in lines:
    spline=line.replace("\n", "").split(",")
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    #some lines have unexpected line breaks which mess up the output (the last item in the list is the tweet, not the ER)

    try:
        #important to convert to float!!
        number=float(spline[6])
        if (number <= 0.10):
            erlist.append(number)
    except:
        print ("Skipping")

#print (erlist)

cp.get_histogram(erlist)

'''
'''

"""Get lexical diversity"""

#cp.lexical_diversity()


"""Get average number of words"""

#cp.average_words()

"""Get average"""


#average = cp.get_average(user_list)
#ext.printcsv_all(average,'av')


'''
"""Plot graph"""

'''

lines = open('output_av.csv','r').readlines()

x = []
y = []

for line in lines:
    spline=line.replace("\n", "").split(",")
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    x.append(spline[0])
    y.append(spline[1])

x = list(map(float,x))
y = list(map(float,y))

plt.plot(x,y,'*')
plt.show()

'''

