

from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import csv
import os
import sys


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

consumer_key = api_dict["API_key"]
consumer_secret = api_dict["API_secret"]

access_token = api_dict["Access_token"]
access_token_secret = api_dict["Access_token_secret"]


hashtaglist = ['ssf15']

csvfile=open('output/output_sydsciencefest_stream.csv','a', newline='')
csvtweets = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)

class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, status):


        data=json.loads(status)

        if 'created_at' in data:
            print ([data['created_at'],data['text']])
            csvtweets.writerow([data['created_at'], data['retweet_count'], data['favorite_count'], data['text'].replace('\n', ' ').replace(',', ' ')])
        #flush method writes data in buffer directly to the target file (real-time data writing to file)
            csvfile.flush()


        #for hashtag in data['entities']['hashtags']:

            #if hashtag['text'].lower() in hashtaglist:
                #print(hashtag['text'])
                #print(data['text'])
                #csvtweets.writerow([data['text']])
                #csvfile.flush()
                #csvtweets.writerow(status)

        return True

    def on_error(self, status):
        print(status)

        return True

    def on_timeout(self):
        print('Timeout...')
        return True # To continue listening

if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    #stream.filter(languages=["en"],track=["#"+x for x in hashtaglist])
    stream.filter(languages=["en"],track=['ssf15'])
    #stream.filter(track=hashtaglist)



