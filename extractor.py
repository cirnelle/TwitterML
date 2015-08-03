

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

#users = ["nasa","cern"]

if os.path.exists("/Users/yi-linghwong/GitHub/TwitterML/"):
    maindir = "/Users/yi-linghwong/GitHub/TwitterML/"
elif os.path.exists("/home/yiling/GitHub/GitHub/TwitterML/"):
    maindir = "/home/yiling/GitHub/GitHub/TwitterML/"
else:
    print ("ERROR --> major error")
    sys.exit(1)

if os.path.isfile('../../keys/twitter_api_keys.txt'):
    lines = open('../../keys/twitter_api_keys.txt','r').readlines()


else:
    print ("Path not found")
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


class Extractor():

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



    def is_rate_limit_error(self,e):
        return isinstance(e.reason, list) \
            and e.reason[0:] \
            and 'code' in e.reason[0] \
            and e.reason[0]['code'] == 88


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
        #IMPORTANT: remember to set wait_on_rate_limit parameter to True to avoid rate limiting
        api = tweepy.API(auth, wait_on_rate_limit=True)

        return api

    def user_tweets(self,user,api):

        tweets=tweepy.Cursor(api.user_timeline,id=user, include_rts=False, exclude_replies=True).items(500)

        return tweets


    def hashtag_tweets(self,hashtag,api):

        tweets=tweepy.Cursor(api.search, q=hashtag, lang="en").items(50)

        return tweets


    def gettweets_user(self,user,api):

        rate_limit = api.rate_limit_status()

        remaining = rate_limit['resources']['statuses']['/statuses/user_timeline']['remaining']
        reset_time = rate_limit['resources']['statuses']['/statuses/user_timeline']['reset']

        print (remaining)

        #if remaining >= 10:


        #tweets=api.user_timeline(fromuser)
        #while True:
            #try:

        # create a Cursor instance. Cursor is a class in tweepy to help with pagination (iterate through statuses
        # to get more than 20 tweets from user_timeline)


        tweets = self.user_tweets(user,api)

        #else:

            #print ("Rate limiting. Going to sleep")

            #sleep_time = int(reset_time) - int(time.time())

            #time.sleep(sleep_time)

            #tweets = self.user_tweets(user,api)


        #for i in range(len(tweets)):


        """

        except:
            print ("Entering error")
            #if e == {"errors":[{"message":"Rate limit exceeded","code":88}]}:
                #time.sleep(60*5) #Sleep for 5 minutes
            #else:
                #print (e)



            except tweepy.error.TweepError as e:
                print ("Entering error")
                if not self.is_rate_limit_error(e):
                    raise e
                else:
                    print ('Rate limit error')
                    time.sleep(60*5)

            """



        fulltweets=[]

        for tweet in tweets:

            #tweet=tweets[tweet]

            #dumps serialises strings into JSON (which is very similar to python's dict)
            json_str= json.dumps(tweet._json)

            #loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
            data=json.loads(json_str)




            #add the new tweets to a list
            fulltweets.append([user, data['created_at'], data['text'].replace('\n', ' '), data['user']['followers_count'], data['user']['friends_count'], data['retweet_count'], data['favorite_count']])

            ##IMPORTANT: the 'followers_count' key is in a dictionary (called 'user') within a dictionary!
                #fulltweets.append([data['user']['followers_count'], data['retweet_count']])





        #fulltweets is a list in a list, i.e. [[text, rt count, etc], [text2, rt count2, etc]]
        return fulltweets

            #print (data['text'])
            #print ("Retweets ",data['retweet_count'])
            #print ("Favorited ",data['favorite_count'])
            #print (data['created_at'])
            #print ("------------")
            #print (data)



    def gettweets_hashtag(self,hashtag,api):

        rate_limit = api.rate_limit_status()


        remaining = rate_limit['resources']['statuses']['/statuses/user_timeline']['remaining']
        reset_time = rate_limit['resources']['statuses']['/statuses/user_timeline']['reset']

        print (remaining)

        #if remaining >= 10:

        #self.requestlimit()

        tweets=self.hashtag_tweets(hashtag,api)

        #else:

            #print ("Rate limiting. Going to sleep")

            #sleep_time = int(reset_time) - int(time.time())

            #time.sleep(sleep_time)

            #tweets=self.hashtag_tweets(hashtag,api)

        fulltweets=[]

        for tweet in tweets:

            #tweet=tweets[tweet]

            #dumps serialises strings into JSON (which is very similar to python's dict)
            json_str= json.dumps(tweet._json)

            #loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
            data=json.loads(json_str)


            #add the new tweets to a list
            fulltweets.append([data['created_at'], data['text'].replace('\n', ' '), data['retweet_count'], data['favorite_count']])

        return fulltweets


    def printcsv(self,all_tweets,filename):


        with open(maindir+'/output/output_'+filename+'.csv', 'w', newline='') as csvfile:
            csvtweets = csv.writer(csvfile, delimiter='|',quoting=csv.QUOTE_MINIMAL)

            for al in all_tweets:
                csvtweets.writerow(al)
                csvfile.flush()
                os.fsync(csvfile.fileno())




    def printcsv_all(self,all_tweets,name):
        with open(maindir+'/output/output_'+name+'.csv','w', newline='') as csvfile:
            csvtweets = csv.writer(csvfile, delimiter='|',quoting=csv.QUOTE_MINIMAL)

            for al in all_tweets:
                csvtweets.writerow(al)
                csvfile.flush()
                os.fsync(csvfile.fileno())





#ext= Extractor()
#auth = ext.loadtokens()
#api = ext.connectToAPI(auth)

#full_tweets = []

#for user in users:

    #use extend instead of append to add the second list to the first one!
    #full_tweets.extend(ext.gettweets_user(user,api))
    #ext.printcsv(full_tweets,user)

#ext.printcsv_all(full_tweets)



#print (ext.gettweets_hashtag(hashtaglist,api))

#for ht in hashtaglist:

    #full_tweets = ext.gettweets_hashtag(ht,api)
    #ext.printcsv(full_tweets,ht)

