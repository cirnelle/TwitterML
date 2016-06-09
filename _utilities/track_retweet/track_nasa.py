__author__ = 'yi-linghwong'

import os
import sys
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time
import json


################
# get retweet growth count
################

while os.stat('retweet_files/nasa_id.csv').st_size==0:
    time.sleep(5)
    print ("empty file")

if (os.stat('retweet_files/nasa_id.csv').st_size==0) == False:

    ###################
    # create dict with api keys
    ###################

    if os.path.isfile('../../../../keys/twitter_api_keys.txt'):
        lines = open('../../../../keys/twitter_api_keys.txt', 'r').readlines()

    else:
        print("Path not found")
        sys.exit(1)

    api_dict = {}

    for line in lines:
        spline = line.replace("\n", "").split()

        api_dict[spline[0]] = spline[1]

    apikey = api_dict["API_key"]
    apisecret = api_dict["API_secret"]

    AccessToken = api_dict["Access_token"]
    AccessTokenSecret = api_dict["Access_token_secret"]

    auth = tweepy.OAuthHandler(apikey, apisecret)
    auth.set_access_token(AccessToken, AccessTokenSecret)

    print("Connecting to twitter API...")

    api = tweepy.API(auth, wait_on_rate_limit=True)

    #################
    # get retweet count
    #################

    print ()

    print ("Not empty!")

    lines = open('retweet_files/esa_id.csv','r').readlines()

    tweet_id = []

    for line in lines:
        spline = line.replace('\n' ,'').split(',')
        tweet_id.append(spline[2])
        print ("Got id for tweet...")

    print (tweet_id)

    retries = 5


    for i in range(5760): # 24 hours, every 15 seconds

        for id in tweet_id:

            print (id)

            tweets = []

            tweet = api.get_status(id=id)

            # print (tweet)

            for r in range(retries):

                try:

                    rate_limit = api.rate_limit_status()

                    remaining = rate_limit['resources']['statuses']['/statuses/user_timeline']['remaining']
                    reset_time = rate_limit['resources']['statuses']['/statuses/user_timeline']['reset']

                    print(remaining)

                    # dumps serialises strings into JSON (which is very similar to python's dict)
                    json_str = json.dumps(tweet._json)

                    # loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
                    data = json.loads(json_str)

                    #################
                    # check if media exists, and which type
                    #################

                    if 'extended_entities' in data:

                        if 'media' in data['extended_entities']:

                            if data['extended_entities']['media'] != []:

                                length = len(data['extended_entities']['media'])

                                for n in range(length):
                                    type = data['extended_entities']['media'][n]['type']


                    elif 'entities' in data:

                        if 'urls' in data['entities']:

                            if (data['entities']['urls'] != []):

                                length = len(data['entities']['urls'])

                                for n in range(length):

                                    if (data['entities']['urls'][n]['display_url'].startswith('youtu')):
                                        type = 'video'
                                        break

                                    elif (data['entities']['urls'][n]['display_url'].startswith('vine')):
                                        type = 'video'
                                        break

                                    elif (data['entities']['urls'][n]['display_url'].startswith('amp.twimg')):
                                        type = 'video'
                                        break

                                    elif (data['entities']['urls'][n]['display_url'].startswith('snpy.tv')):
                                        type = 'video'
                                        break

                                    elif (data['entities']['urls'][n]['display_url'].startswith('vimeo')):
                                        type = 'video'
                                        break

                                    else:
                                        type = 'no_media'

                            else:
                                type = 'no_media'

                        else:
                            type = 'no_media'

                    else:
                        type = 'no_media'

                    ################
                    # append list of parameters to tweet list
                    ################

                    tweets.append([data['user']['screen_name'], data['created_at'], str(time.time()), str(time.localtime(time.time())), data['id_str'],
                                   str(data['user']['followers_count']), str(data['user']['friends_count']),
                                   str(data['retweet_count']), str(data['favorite_count']), 'has_' + str(type),
                                   data['text'].replace('\n', ' ').replace('\r', '').replace('\t', ' ').replace(',', ' ')])

                    #################
                    # write (append) data to file for each user
                    #################

                    f = open('retweet_files/esa_retweet.csv', 'a')

                    for t in tweets:
                        f.write(','.join(t) + '\n')

                    f.close()

                    break

                except Exception as e:
                    print('Failed: ' + str(e))
                    time.sleep(5)

        time.sleep(15)
