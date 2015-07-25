__author__ = 'yi-linghwong'

from extractor import Extractor
import os
import numpy as np
from matplotlib import pyplot as plt
import time



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


            for t in full_tweets:
                engrate = (np.divide(t[3],t[2]))*100

                engrate_list.append([user, t[1], engrate])

        return engrate_list


    def get_histogram(self,engrate_list):

        #list with only engagement rate
        #er_list = []

        #for er in engrate_list:

            #er_list.append(er[2])

        #print (er_list)

        plt.hist(engrate_list,bins=20)
        #need block=True to keep plot opened
        plt.xlabel("Engagement rate")
        plt.show(block=True)



cp = Compute()

'''
Get engagement rate
'''


engratelist = cp.get_eng_rate(user_list)
ext.printcsv_all(engratelist,'engrate')

'''

if os.path.isfile('output_engrate.csv'):
  lines = open('output_engrate.csv','r').readlines()

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
        number=float(spline[len(spline)-1])
        erlist.append(number)
    except:
        print ("Skipping")

#print (erlist)

cp.get_histogram(erlist)



Get average
'''
#average = cp.get_average(user_list)
#ext.printcsv_all(average,'av')

'''
Plot graph


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