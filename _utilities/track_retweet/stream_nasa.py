__author__ = 'yi-linghwong'

import os
import sys
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time
import json

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
    # creates a list with key and value. Split splits a string at the space and stores the result in a list

    api_dict[spline[0]] = spline[1]

consumer_key = api_dict["API_key"]
consumer_secret = api_dict["API_secret"]

access_token = api_dict["Access_token"]
access_token_secret = api_dict["Access_token_secret"]


class StdOutListener(StreamListener):

    def __init__(self):

        print ("Waiting for data ...")

    ##############
    # defines a class that inherits from tweepy's StreamListener class
    ##############

    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, status):

        data=json.loads(status)

        for n in range(5):

            try:

                if 'created_at' in data:

                    # exclude retweets from stream

                    if 'retweeted_status' not in data:

                        if (data['in_reply_to_status_id_str'] == None) and (data['in_reply_to_user_id'] == None) and (('@ESA' and '@esa' and '@Esa') not in data['text']):

                            print(data)
                            print ([data['created_at'],data['text']])

                            tweet = [data['user']['screen_name'],data['created_at'],data['id_str'],str(data['user']['followers_count']),str(data['user']['friends_count']),str(data['retweet_count']),str(data['favorite_count']),data['text'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')]

                            f = open(path_to_store_streamed_tweets,'a')

                            f.write(','.join(tweet)+'\n')

                            f.close()

                            print("gotten data for NASA, quitting...")

                            sys.exit()

                            break

                            #csvtweets.writerow([data['user']['screen_name'], data['id_str'], data['created_at'], data['user']['followers_count'], data['user']['friends_count'], data['retweet_count'], data['favorite_count'], data['text'].replace('\n', ' ').replace(',', ' ')])
                            #flush method writes data in buffer directly to the target file (real-time data writing to file)
                            #csvfile.flush()


                        else:
                            break

                    else:
                        break

                else:
                    break

            except Exception as e:
                print('Failed: ' + str(e))
                time.sleep(10)


        return True


    def write_to_file(self,tweets):

        print ("Writing data to file ...")

        f = open(path_to_store_streamed_tweets,'a')

        for t in tweets:
            f.write(','.join(t)+'\n')
            f.close()



        return


    def on_error(self, status):

        print(status)

        return True

    def on_timeout(self):
        print('Timeout...')
        return True # To continue listening


###############
# variables
###############

path_to_store_streamed_tweets = 'retweet_files/esa_id.csv'
hashtaglist = ['coffee','science'] #amounts to logical OR

if __name__ == '__main__':

    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)

    stream.filter(languages=["en"],follow=['21436960'], async=True)
    # NASA, neiltyson, Cmdr_Hadfield, SpaceX, SKA_telescope, Space_Station, esa, StarTalkRadio
    #stream.filter(languages=["en"], track=hashtaglist, async=True)
