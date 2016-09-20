import os
import sys
import subprocess
from pymongo import MongoClient
from datetime import datetime

client = MongoClient()

db = client.twitter_db

tweets_iterator = db.twitter_collection.find()

tweets = []

for tweet in tweets_iterator:

    tweets.append(tweet['text'])

    #print (tweet['text'])

print (len(tweets))

f = open('test.csv','w')

for t in tweets:
    f.write(t+"\n")

f.close()