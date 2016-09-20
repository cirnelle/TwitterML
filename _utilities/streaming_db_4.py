__author__ = 'yi-linghwong'

import os
import sys
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from pymongo import MongoClient
import json
import subprocess
from twilio.rest import TwilioRestClient

#------------------------------------
# Twitter API
#------------------------------------


if os.path.exists("/Users/yi-linghwong/GitHub/TwitterML/"):
    maindir = "/Users/yi-linghwong/GitHub/TwitterML/"
elif os.path.exists("/home/yiling/GitHub/GitHub/TwitterML/"):
    maindir = "/home/yiling/GitHub/GitHub/TwitterML/"
else:
    print ("ERROR --> major error")
    sys.exit(1)


if os.path.isfile('../../../keys/twitter_api_keys_8.txt'):
    lines = open('../../../keys/twitter_api_keys_8.txt','r').readlines()


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

#-------------------------------
# Twilio API, to send sms alerts
#-------------------------------

ACCOUNT_SID = 'AC817c734cbe1b13d2bbe97940e4efc413'
AUTH_TOKEN = 'd927e3830852ec14b06b302960833645'

twilio_client = TwilioRestClient(ACCOUNT_SID, AUTH_TOKEN)


class listener(StreamListener):

    def __init__(self, start_time=time.time(), time_limit=60):

        self.flag = True

        FNULL = open(os.devnull, 'w')
        mongod = subprocess.Popen(['mongod', '--dbpath', os.path.expanduser('/data/db')], stdout=FNULL, stderr=subprocess.STDOUT)

        self.time = start_time
        self.limit = time_limit


    def on_data(self, status):

        #while (time.time() - self.time) < self.limit:

        for n in range(5):

            try:

                client = MongoClient('localhost', 27017)
                db = client['twitter_iss']
                collection = db['iss_collection']
                data = json.loads(status)

                if 'created_at' in data:

                    # exclude retweets from stream

                    if 'retweeted_status' not in data:
                        print([data['created_at'], data['text']])

                        screen_name = data['user']['screen_name']
                        created_at = data['created_at']
                        tweet_id = data['id_str']
                        followers_count = data['user']['followers_count']
                        friends_count = data['user']['friends_count']
                        retweet_count = data['retweet_count']
                        favourite_count = data['favorite_count']
                        text = data['text'].replace('\n', ' ').replace('\r', '').replace('\t', ' ').replace(',', ' ')

                        tweets = {"screen_name": screen_name, "created_at": created_at, "id_str": tweet_id,
                                  "followers_count": followers_count, "friends_count": friends_count,
                                  "retweet_count": retweet_count, "favourite_count": favourite_count, "text": text,}

                        collection.insert(tweets)

                        self.flag = True

                        break

                    else:
                        break

                else:
                    break


            except BaseException as e:

                print ('failed ondata,', str(e))

                # send SMS alert, flag is set to false after first error to prevent continuous sms sending

                if self.flag:

                    twilio_client.messages.create(
                        to='+61406815706',
                        from_='+61447752987',
                        body='iss failed ondata',
                    )

                    self.flag = False

                time.sleep(5)

        return True #when return False then process finished with exit code 0


    def on_error(self, status):

        print(status)

        twilio_client.messages.create(
            to='+61406815706',
            from_='+61447752987',
            body='iss failed on error',
        )

        return True

    def on_timeout(self):

        print('Timeout...')

        twilio_client.messages.create(
            to='+61406815706',
            from_='+61447752987',
            body='iss timeout',
        )

        return True  # To continue listening


if __name__ == '__main__':

    l = listener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)

    stream.filter(languages=["en"],track=['#iss,space station,soyuz,astronaut'], async=True)
    #stream.filter(languages=["en"], track=hashtaglist, async=True)


