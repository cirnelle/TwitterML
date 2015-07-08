

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


##
# variables
##
querylimit = 180
timewindow = 15
users = ["NASA", "CERN"]

if os.path.isfile('../../keys/twitter_api_keys.txt'):
    lines = open('../../keys/twitter_api_keys.txt','r').readlines()


else:
    print "Path not found"
    sys.exit(1)

api_dict = {}

for line in lines:
    spline=line.replace("\n","").split()
    #creates a list with key and value. Split splits a string at the space and stores the result in a list

    api_dict[spline[0]]=spline[1]

apikey = api_dict["API_key"]
apisecret = api_dict["API_secret"]

AccessToken	= api_dict["Access_token"]
AccessTokenSecret = api_dict["Access_token_secret"]

starttime = time.time()

nrequest = 0

auth = ''


class extractor():

    def __init__(self):
        self.printer("Welcome",1)


        global apikey
        global apisecret

        global querylimit
        global timewindow

        global currenttime

        global AccessToken
        global AccessTokenSecret




    def printer(self,messagew,case):
        #
        # 1: info
        # 2: error
        # 3: warning


        if case == 1:
            print ("Info  ==> ",messagew)

        elif case == 2:
            print ("Error  ==> ",messagew)

        elif case == 3:
            print ("Warning  ==> ",messagew)


    def requestlimit(self):

        global starttime
        global nrequest

        currenttime=time.time()
        delta= currenttime-starttime


         #Reset clock when the 15 mins timewindow is up
        if (delta)>(timewindow*60):
            starttime = currenttime
            self.printer("Resetting clock",1)
            nrequest = 1
            delta = 0

        else:
            nrequest=+1

        #Put program to sleep if more than 180 requests in 15 mins
        if nrequest>querylimit:
            self.printer("Going to sleep for "+ str(timewindow*60-delta) +" seconds", 3)
            time.sleep(delta)



    def loadtokens(self):

        #tokenfile = open("access.txt","o")
        #lines=tokenfile.readlines()

        #access_token=lines[0]
        #token_secret=lines[1]

        #create an OAuthHandler instance
        auth = tweepy.OAuthHandler(apikey, apisecret)
        auth.set_access_token(AccessToken, AccessTokenSecret)


        return auth


    def connectToAPI(self,auth):

        self.printer("connecting to twitter API",1)

        #create an API instance. API is a class in tweepy that provides access to the entire
        #Twitter RESTful API methods.
        api = tweepy.API(auth)

        return api

    def gettweets(self,user,api):

        self.requestlimit()

        #tweets=api.user_timeline(fromuser)

        # create a Cursor instance. Cursor is a class in tweepy to help with pagination (iterate through statuses
        # to get more than 20 tweets from user_timeline)
        tweets=tweepy.Cursor(api.user_timeline,id=user).items(100)

        #for i in range(len(tweets)):

        for tweet in tweets:

            #tweet=tweets[tweet]

            json_str= json.dumps(tweet._json)

            data=json.loads(json_str)
            print (data['text'])
            print ("Retweets ",data['retweet_count'])
            print ("Favorited ",data['favorite_count'])
            print (data['created_at'])
            print ("------------")
                #print (data)



ext= extractor()
auth = ext.loadtokens()
api = ext.connectToAPI(auth)

for user in users:

    ext.gettweets(user,api)