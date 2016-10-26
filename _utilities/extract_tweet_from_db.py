import os
import sys
import subprocess
from pymongo import MongoClient
from datetime import datetime

client = MongoClient()

#----------------------
# create iterator for each database

db = client.twitter_astrobiology

astrobiology_iterator = db.astrobiology_collection.find()

db = client.twitter_nasa

nasa_iterator = db.nasa_collection.find()

db = client.twitter_space

space_iterator = db.space_collection.find()

db = client.twitter_iss

iss_iterator = db.iss_collection.find()

db = client.twitter_planets

planets_iterator = db.planets_collection.find()

db = client.twitter_climate

climate_iterator = db.climate_collection.find()

db = client.twitter_spaceusers

spaceusers_iterator = db.spaceusers_collection.find()

astro_tweets = []
nasa_tweets = []
space_tweets = []
iss_tweets = []
planets_tweets = []
climate_tweets = []

spaceusers_tweets = []

#-----------------------
# iterate through tweets and append to list

for tweet in astrobiology_iterator:

    astro_tweets.append([tweet['screen_name'], tweet['created_at'], tweet['id_str'], tweet['in_reply_to_user'],
                         str(tweet['in_reply_to_status_id']),
                         str(tweet['followers_count']), str(tweet['friends_count']), str(tweet['retweet_count']),
                         str(tweet['favourite_count']), tweet['text']])

    #print (tweet['text'])


print (len(astro_tweets))

for tweet in nasa_iterator:

    nasa_tweets.append([tweet['screen_name'], tweet['created_at'], tweet['id_str'], tweet['in_reply_to_user'],
                        str(tweet['in_reply_to_status_id']),
                        str(tweet['followers_count']), str(tweet['friends_count']), str(tweet['retweet_count']),
                        str(tweet['favourite_count']), tweet['text']])

    #print (tweet['text'])


print (len(nasa_tweets))

for tweet in space_iterator:

    space_tweets.append([tweet['screen_name'], tweet['created_at'], tweet['id_str'], tweet['in_reply_to_user'],
                         str(tweet['in_reply_to_status_id']),
                         str(tweet['followers_count']), str(tweet['friends_count']), str(tweet['retweet_count']),
                         str(tweet['favourite_count']), tweet['text']])

    # print (tweet['text'])


print(len(space_tweets))

for tweet in iss_iterator:

    iss_tweets.append([tweet['screen_name'], tweet['created_at'], tweet['id_str'], tweet['in_reply_to_user'],
                       str(tweet['in_reply_to_status_id']),
                       str(tweet['followers_count']), str(tweet['friends_count']), str(tweet['retweet_count']),
                       str(tweet['favourite_count']), tweet['text']])

    # print (tweet['text'])


print(len(iss_tweets))

for tweet in planets_iterator:

    planets_tweets.append([tweet['screen_name'], tweet['created_at'], tweet['id_str'], tweet['in_reply_to_user'],
                           str(tweet['in_reply_to_status_id']),
                           str(tweet['followers_count']), str(tweet['friends_count']), str(tweet['retweet_count']),
                           str(tweet['favourite_count']), tweet['text']])

    # print (tweet['text'])


print(len(planets_tweets))

for tweet in climate_iterator:

    climate_tweets.append([tweet['screen_name'], tweet['created_at'], tweet['id_str'], tweet['in_reply_to_user'],
                           str(tweet['in_reply_to_status_id']),
                           str(tweet['followers_count']), str(tweet['friends_count']), str(tweet['retweet_count']),
                           str(tweet['favourite_count']), tweet['text']])

    # print (tweet['text'])


print(len(climate_tweets))

for tweet in spaceusers_iterator:

    spaceusers_tweets.append([tweet['screen_name'], tweet['created_at'], tweet['id_str'], tweet['in_reply_to_user'],
                              str(tweet['in_reply_to_status_id']),
                              str(tweet['followers_count']), str(tweet['friends_count']), str(tweet['retweet_count']),
                              str(tweet['favourite_count']), tweet['text']])

    # print (tweet['text'])


print(len(spaceusers_tweets))

#---------------------------------
# write to file

f = open('../tweets/streaming/db/astrobiology.csv','w')

for t in astro_tweets:
    f.write(','.join(t)+"\n")

f.close()

f = open('../tweets/streaming/db/nasa.csv','w')

for t in nasa_tweets:
    f.write(','.join(t)+"\n")

f.close()

f = open('../tweets/streaming/db/space.csv','w')

for t in space_tweets:
    f.write(','.join(t)+"\n")

f.close()

f = open('../tweets/streaming/db/iss.csv','w')

for t in iss_tweets:
    f.write(','.join(t)+"\n")

f.close()

f = open('../tweets/streaming/db/planets.csv','w')

for t in planets_tweets:
    f.write(','.join(t)+"\n")

f.close()

f = open('../tweets/streaming/db/climate.csv','w')

for t in planets_tweets:
    f.write(','.join(t)+"\n")

f.close()

f = open('../tweets/streaming/db/space_users.csv','w')

for t in spaceusers_tweets:
    f.write(','.join(t)+"\n")

f.close()