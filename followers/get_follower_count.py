#!/usr/local/bin/python3

##########################
#
# version 1.0
# comments: - initial creation of the class
#
###########################
__author__ = 'yi-linghwong'

##
# imports
##
import time
import tweepy
import json
import os
import sys
import csv
from tweepy.streaming import StreamListener


##
# variables
##
querylimit = 180
timewindow = 15

if os.path.isfile('../user_list/user_MASTER.csv'):
  lines = open('../user_list/user_MASTER.csv','r').readlines()

else:
    print ("Path not found")
    sys.exit(1)

users = []

for line in lines:
    spline=line.replace('\n','').split(',')
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    users.append(spline[0])



#users = ["nasa","cern"]

if os.path.isfile('/Users/yi-linghwong/keys/twitter_api_keys.txt'):
    lines = open('/Users/yi-linghwong/keys/twitter_api_keys.txt','r').readlines()


else:
    print ("Path not found")
    sys.exit(1)

api_dict = {}

for line in lines:
    spline=line.replace("\n","").split()
# creates a list with key and value. Split splits a string at the space and stores the result in a list

    api_dict[spline[0]]= spline[1]

apikey = api_dict["API_key"]
apisecret = api_dict["API_secret"]

AccessToken	= api_dict["Access_token"]
AccessTokenSecret = api_dict["Access_token_secret"]

starttime = time.time()

nrequest = 0

auth = ''

class Extractor():

    def __init__(self):
        self.printer("Welcome", 1)

        global apikey
        global apisecret

        global querylimit
        global timewindow

        global currenttime

        global AccessToken
        global AccessTokenSecret

    def printer(self, messagew, case):
        #
        # 1: info
        # 2: error
        # 3: warning


        if case == 1:
            print("Info  ==> ", messagew)

        elif case == 2:
            print("Error  ==> ", messagew)

        elif case == 3:
            print("Warning  ==> ", messagew)

    def requestlimit(self):

        global starttime
        global nrequest

        currenttime = time.time()
        delta = currenttime - starttime


        # Reset clock when the 15 mins timewindow is up
        if (delta) > (timewindow * 60):
            starttime = currenttime
            self.printer("Resetting clock", 1)
            nrequest = 1
            delta = 0

        else:
            nrequest = +1

        # Put program to sleep if more than 180 requests in 15 mins
        if nrequest > querylimit:
            self.printer("Going to sleep for " + str(timewindow * 60 - delta) + " seconds", 3)
            time.sleep(delta)

    def loadtokens(self):

        # tokenfile = open("access.txt","o")
        # lines=tokenfile.readlines()

        # access_token=lines[0]
        # token_secret=lines[1]

        # create an OAuthHandler instance
        auth = tweepy.OAuthHandler(apikey, apisecret)
        auth.set_access_token(AccessToken, AccessTokenSecret)

        return auth

    def connectToAPI(self, auth):

        self.printer("connecting to twitter API", 1)

        # create an API instance. API is a class in tweepy that provides access to the entire
        # Twitter RESTful API methods.
        api = tweepy.API(auth, wait_on_rate_limit=True)

        return api

    def get_follower_count(self, user, api):

        rate_limit = api.rate_limit_status()

        remaining = rate_limit['resources']['statuses']['/statuses/user_timeline']['remaining']
        reset_time = rate_limit['resources']['statuses']['/statuses/user_timeline']['reset']

        #print ("Rate limit remaining "+str(remaining))

        try:

            user_info = api.get_user(user)
            follcount = user_info.followers_count

        except:
            print ("error for "+str(user))
            follcount = 'nan'


        # create a Cursor instance. Cursor is a class in tweepy to help with pagination (iterate through statuses
        # to get more than 20 tweets from user_timeline)
        # tweets=tweepy.Cursor(api.user_timeline,id=user, include_rts=False, exclude_replies=True).items(200)

        # for i in range(len(tweets)):

        return follcount

    def printcsv(self, date, follcount, user):

        with open('follcount_'+user+'.csv', 'a', newline='') as csvfile:
            csvtweets = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvtweets.writerow([follcount, date])



ext = Extractor()
auth = ext.loadtokens()
api = ext.connectToAPI(auth)

#ext.get_follower_count('nasa', api)
now = time.strftime("%c")

#print ("Current time %s"  % now )

for user in users:
    follcount = ext.get_follower_count(user, api)
    ext.printcsv(follcount, now, user)
    time.sleep(1)