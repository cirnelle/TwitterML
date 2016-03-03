__author__ = 'yi-linghwong'

import os
import sys
import tweepy
import json

if os.path.isfile('../../../keys/twitter_api_keys.txt'):
    lines = open('../../../keys/twitter_api_keys.txt','r').readlines()


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

auth = tweepy.OAuthHandler(apikey, apisecret)
auth.set_access_token(AccessToken, AccessTokenSecret)



api = tweepy.API(auth, wait_on_rate_limit=True)

#tweets=tweepy.Cursor(api.user_timeline,id='nasa', include_rts=False, exclude_replies=True).items(1)

tweet=api.get_status(id=705364886784618496)
print (tweet)

fulltweets = []



#dumps serialises strings into JSON (which is very similar to python's dict)
json_str= json.dumps(tweet._json)

#loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
data=json.loads(json_str)


if 'extended_entities' in data:

    if 'media' in data['extended_entities']:

        print ("Extended")

        type = data['extended_entities']['media'][0]['type']


    fulltweets.append(type)
    #fulltweets.append(['nasa', data['created_at'], data['id'], data['entities']['media']['type'], data['user']['followers_count'], data['user']['friends_count'], data['retweet_count'], data['favorite_count'],data['text'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])

    print (fulltweets)


elif 'entities' in data:

    if 'urls' in data['entities']:

        if (data['entities']['urls'] != []):

            if (data['entities']['urls'][0]['display_url'].startswith('youtu')):
                type = 'video'

            else:
                type = 'no_media'

        else:
            type = 'no_media'

    else:
        type = 'no_media'


    fulltweets.append(type)
    #fulltweets.append(['nasa', data['created_at'], data['id'], data['entities']['media']['type'], data['user']['followers_count'], data['user']['friends_count'], data['retweet_count'], data['favorite_count'],data['text'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])


    print (fulltweets)



