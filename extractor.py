##########################
#
# version 1.0
# comments: - initial creation of the class
#
###########################
__author__ = 'gling'

##
# imports
##
import time
import tweepy
import json


##
# variables
##
querielimit = 180
timewindow = 15


apikey = ""
apisecret = ""

AccessToken	= ""
AccessTokenSecret = ""

starttime = 0
currentime = 0
nrequest = 0

auth = ''


class extractor():

    def __init__(self):
        self.printer("Welcome",1)

        global apikey
        global apisecret

        global querielimit
        global timewindow

        global currenttime

        global AccessToken
        global AccessTokenSecret




    def printer(self,messagew,switch):
        #
        # 1: info
        # 2: error
        # 3: warning


        if switch == 1:
            print ("Info  ==>",messagew)

        elif switch == 2:
            print ("Error  ==>",messagew)

        elif switch == 3:
            print ("Warning  ==>",messagew)


    def requestlimit(self):

        global starttime
        global nrequest

        currenttime=time.time()
        delta= currenttime-starttime

        if nrequest>querielimit:
            self.printer("Going to sleep for "+delta+" seconds", 3)
            time.sleep(delta)

        if (delta)>(timewindow*60):
            starttime = currenttime
            self.printer("Resetting clock",1)
            nrequest=1
        else:
            nrequest=+1





    def loadtokens(self):

        #tokenfile = open("access.txt","o")
        #lines=tokenfile.readlines()

        #access_token=lines[0]
        #token_secret=lines[1]

        auth = tweepy.OAuthHandler(apikey, apisecret)
        auth.set_access_token(AccessToken, AccessTokenSecret)

        return auth


    def connectToAPI(self,auth):

        self.printer("connecting to twitter API",1)

        api = tweepy.API(auth)

        return api

    def gettweetst(self,fromuser,api):

        self.requestlimit()

        tweets=api.user_timeline(fromuser)

        for i in range(len(tweets)):

            tweet=tweets[i]

            json_str= json.dumps(tweet._json)

            data=json.loads(json_str)
            print (data['text'])
            print ("Retweets ",data['retweet_count'])
            print ("Favorited ",data['favorite_count'])
            print (data['created_at'])
            print ("------------")
            print (data)






ext= extractor()
auth =ext.loadtokens()
api = ext.connectToAPI(auth)
ext.gettweetst("@NASA",api)