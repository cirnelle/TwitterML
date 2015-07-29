__author__ = 'yi-linghwong'

from extractor import Extractor
import os
import numpy
from matplotlib import pyplot as plt

"""

if os.path.isfile('users_temp.txt'):
  lines = open('users_temp.txt','r').readlines()

else:
    print ("Path not found")
    sys.exit(1)

user_list=[]

for line in lines:
    spline=line.replace("\n", "").split(",")
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    user_list.append(spline[0])

"""

user_list=['nasa']

ext= Extractor()
auth = ext.loadtokens()
api = ext.connectToAPI(auth)

class UserTweets():

    def get_user_tweets(self,users):

        for user in users:

            full_tweets = ext.gettweets_user(user,api)
            ext.printcsv(full_tweets,user)


    def get_all_user_tweets(self,users):

        full_tweets = []

        for user in users:

            #use extend instead of append to add the second list to the first one!
            full_tweets.extend(ext.gettweets_user(user,api))



            #ext.printcsv(full_tweets,user)

        return full_tweets







'''
Get tweets per user per file
'''

ut = UserTweets()
#ut.get_user_tweets(user_list)

'''
Get tweets all users in one file
'''

fulltweets = ut.get_all_user_tweets(user_list)
ext.printcsv_all(fulltweets,'all')

'''
Get engagement rate
'''
#engratelist = ut.get_eng_rate(user_list)
#ext.printcsv_all(engratelist,'engrate')


'''
Get average
'''
#average = ut.get_average(user_list)
#ext.printcsv_all(average,'av')

'''
Plot graph


lines = open('output_av.csv','r').readlines()

x = []
y = []

for line in lines:
    spline=line.replace("\n", "").split(",")
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    x.append(spline[0])
    y.append(spline[1])

x = list(map(float,x))
y = list(map(float,y))

plt.plot(x,y,'*')
plt.show()


'''
