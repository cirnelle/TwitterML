

from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json


hashtaglist = ['pluto']

class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, status):


        data=json.loads(status)


        for hashtag in data['entities']['hashtags']:

            if hashtag['text'].lower() in hashtaglist:
                print(hashtag['text'])
                print(data['text'])

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
    stream.filter(track=["#"+x for x in hashtaglist])
    #stream.filter(track=hashtaglist)

