__author__ = 'yi-linghwong'

##############
# Script to process list of influencers,
# get most mentioned, most followed, top retweeted, etc
##############

import sys
import os
import json
from collections import Counter
from extractor import Extractor

#lines = open('temp.txt', 'r').readlines()
lines = open('output/education/output_aussieED.csv', 'r').readlines()

users=[]
mentioned=[]
tweets=[]

for line in lines:

    ###Get a list of all users (twitterers)##
    spline=line.replace('\n','').split(',')
    users.append(spline[0])

    ###Get a list of all users who were mentioned (e.g. Retweeted)##
    l = [word.strip("@").strip(':') for word in line.replace('\n','').split() if word.startswith("@")]
    mentioned.extend(l)

    if (len(spline)==8):

        tweets.append(spline)

most_mentioned=Counter(mentioned).most_common(50000)

user_count=Counter(users).most_common(5000)


temp=[]
temp2=[]
top_RT=[]
most_followed=[]

tweets.sort(key=lambda x:int(x[5]), reverse=True)

##Create list of top retweeted tweets##
for t in tweets:
    if (t[7] not in temp) and (len(top_RT) < 20):
        temp.append(t[7])

        top_RT.append(t)


tweets.sort(key=lambda x:int(x[3]), reverse=True)


##Create list of most followed users##
for t in tweets:
    if (t[0] not in temp2) and (len(most_followed) < 200):
        temp2.append(t[0])
        most_followed.append([t[0], t[3]])

ext= Extractor()
auth = ext.loadtokens()
api = ext.connectToAPI(auth)

id_list=[]

print ("Getting ID of most retweeted tweets...")
## Get the ID of the most retweeted tweets and put them in a list##
## If tweet starts with 'RT', meaning it is a retweet, then get the ID of the original tweet ##
for t in top_RT:

    if (t[7].startswith('RT')) or (t[7].startswith('"RT')):

        tweet=api.get_status(id=int(t[1]))

        #dumps serialises strings into JSON (which is very similar to python's dict)
        json_str= json.dumps(tweet._json)

        #loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
        data=json.loads(json_str)

        ## get id of original tweet ##
        id_list.append(data['retweeted_status']['id'])

    else:
        id_list.append(t[1])

retweeter_list=[]

print ("Getting list of retweeters and their follower count...")


## Get the list of the retweeters and their follower count ##
for id in id_list:

    ## the api limits to only 100 most recent retweets ##
    tweets=api.retweets(id=id,count=200)

    rl=[]

    for t in tweets:

        #dumps serialises strings into JSON (which is very similar to python's dict)
        json_str= json.dumps(t._json)

        #loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
        data=json.loads(json_str)


        rl.append([data['user']['followers_count'], data['user']['screen_name']])

    ## sort the list on the followers count ##
    rl.sort(key=lambda x:int(x[0]), reverse=True)

    ## convert list of lists into list of tuples ##
    rlist = [tuple(l) for l in rl]
    retweeter_list.append(rlist)

## get list with only handles of retweeters ##
rt_handles=[]
for rl in retweeter_list:
    l1=[r[1] for r in rl]
    rt_handles.extend(l1)

retweeter_counts=Counter(rt_handles).most_common(20)

print (retweeter_counts)


##flatten retweeter list, which is a list of lists##
retweeter_list_flat = [item for sublist in retweeter_list for item in sublist]

retweeter_list_flat=set(retweeter_list_flat) ##remove duplicates


## get list of top retweeters and their follower count ##

top_retweeter=[]
for rc in retweeter_counts:
    l2 = [r for r in retweeter_list_flat if rc[0]==r[1]]
    top_retweeter.extend([l2, rc[1]])

print (top_retweeter)

## get list of retweeters with the most followers ##

retweeter_follower=list(retweeter_list_flat)
retweeter_follower.sort(key=lambda x:int(x[0]), reverse=True)
print (retweeter_follower[:10])

influencer=[]
for uc in user_count[:20]:
    influencer.append(uc[0])

for mm in most_mentioned[:20]:
    influencer.append(mm[0])

for mf in most_followed[:20]:
    influencer.append(mf[0])

for rc in retweeter_counts:
    influencer.append(rc[0])

for rf in retweeter_follower[:20]:
    influencer.append(rf[1])

print (influencer)
print (len(influencer))

influencer_counts=Counter(influencer).most_common(10)

print (influencer_counts)


"""Print output to files"""

f=open('output/education/aussieED_most_mentioned.csv', 'w')

for mm in most_mentioned:
    f.write(str(mm)+'\n')

f.close()

f=open('output/education/aussieED_top_users.csv', 'w')

for uc in user_count:
    f.write(str(uc)+'\n')

f.close()

f=open('output/education/aussieED_most_followed.csv', 'w')

for mf in most_followed:
    f.write(str(mf)+'\n')

f.close()

f=open('output/education/aussieED_most_retweeted.csv', 'w')

for tr in top_RT:
    f.write(str(tr)+'\n')

f.close()

f=open('output/education/aussieED_retweeters.csv', 'w')

for rt in retweeter_list:
    f.write(str(rt)+'\n')

f.close()

f=open('output/education/aussieED_top_retweeters.csv', 'w')

for tr in top_retweeter:
    f.write(str(tr)+'\n')

f.close()

f=open('output/education/aussieED_retweeters_by_followers.csv', 'w')

for rf in retweeter_follower[:20]:
    f.write(str(rf)+'\n')

f.close()

f=open('output/education/aussieED_influencers.csv', 'w')

for i in influencer_counts:
    f.write(str(i)+'\n')

f.close()







