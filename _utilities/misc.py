__author__ = 'yi-linghwong'

import os
import sys

#-------------------------------
# get engrate from excel and then insert into csv (maas data)

# lines1 = open('../output/engrate/maas/engrate_from_excel.csv','r').readlines()
# lines2 = open('../tweets/maas/raw_maasmuseum.csv','r').readlines()
#
# id_dict = {}
#
# for line in lines1:
#     spline = line.rstrip('\n').split(',')
#
#     id_dict[spline[0]] = spline[1]
#
# tweets = []
#
# for line in lines2:
#     spline = line.rstrip('\n').split(',')
#     key = spline[2]
#
#     if key in id_dict:
#         engrate = id_dict[key]
#         spline.insert(7,engrate)
#         tweets.append(spline)
#
#     else:
#         print ("key does not exist")
#         print (key)
#
# print (len(tweets))
#
# f = open('../output/engrate/maas/maasmuseum/engrate_maasmuseum_raw.csv','w')
#
# for t in tweets:
#     f.write(','.join(t)+'\n')
#
# f.close()

#-------------------------------------
# get a certain number of tweets from the LER list

# lines = open('../output/engrate/maas/labelled_maas.csv','r').readlines()
#
# tweets = []
# ler = []
#
# for line in lines:
#     spline = line.rstrip('\n').split(',')
#
#     if spline[1] == 'HER':
#         tweets.append(spline)
#
#     if spline[1] == 'LER':
#         if len(ler) < 279:
#             tweets.append(spline)
#             ler.append(spline)
#
#         else:
#             pass
#
# print (len(tweets))
#
# f = open('../output/engrate/maas/labelled_maas_s.csv','w')
#
# for t in tweets:
#     f.write(','.join(t)+'\n')
#
# f.close()


#-------------------------------------
# get only sydneyobs or maasmuseum tweets by id

# lines1 = open('../tweets/maas/maasmuseum_id.txt','r').readlines()
# lines2 = open('../tweets/maas/raw_maas.csv','r').readlines()
#
# ids = []
#
# for line in lines1:
#     spline = line.rstrip('\n')
#     ids.append(spline)
#
# print (len(ids))
#
# tweets = []
#
# for line in lines2:
#     spline = line.rstrip('\n').split(',')
#
#     if spline[2] in ids:
#         tweets.append(spline)
#
# print (len(tweets))
#
# f = open('../tweets/maas/raw_maasmuseum.csv','w')
#
# for t in tweets:
#     f.write(','.join(t)+'\n')
#
# f.close()

###############
# extrapolate between two dates for follower count
###############

import time
import matplotlib.pyplot as plt
import pylab
import numpy as np
from scipy.stats import linregress

lines = open('../user_list/user_TEMP.csv','r').readlines()

users = []

for line in lines:
    spline = line.rstrip('\n').split(',')
    users.append(spline[0])

print (len(users))


for u in users:

    lines = open('../followers/temp/follcount_'+u+'.csv','r').readlines()

    follcount_old = []
    new_list = []

    for line in lines:
        spline = line.rstrip('\n').split(',')
        if spline != ['']:
            follcount_old.append(spline)

    length = len(follcount_old)

    follcount_old.insert(length - 5,['Wed Sep  7 06:00:01 2016','0'])
    # length = len(follcount_old)
    # follcount_old.insert(length - 2,['Fri Jul 10 16:12:47 2016','0'])
    # length = len(follcount_old)
    # follcount_old.insert(length - 3,['Fri Jul  9 16:12:47 2016','0'])

    target_dates = []

    date_epoch = []
    follcount = []

    for line in lines:
        spline = line.rstrip('\n').split(',')

        if spline != ['']:

            d1 = spline[0].replace('\n', '').split(' ')


            if len(d1) == 6:
                d1.remove(d1[2])

            if (d1[1] == 'Sep' and d1[4] == '2016'):

                if d1[2] == '6':
                    date_s = d1[1] + ' ' + d1[2] + ' ' + d1[4]

                    t1 = time.strptime(date_s, '%b %d %Y')
                    t_epoch = time.mktime(t1)
                    date_epoch.append(t_epoch)
                    follcount.append(float(spline[1]))

                    target_dates.append(spline)

                if d1[2] == '8':
                    date_s = d1[1] + ' ' + d1[2] + ' ' + d1[4]

                    t1 = time.strptime(date_s, '%b %d %Y')
                    t_epoch = time.mktime(t1)
                    date_epoch.append(t_epoch)
                    follcount.append(float(spline[1]))

                    target_dates.append(spline)

    print (follcount)
    sys.exit()
    slope = linregress(date_epoch, follcount)[0]
    t_max = date_epoch[1]
    foll_count_max = follcount[1]

    if slope < 0:
        slope = 0.0

    for fo in follcount_old:
        date_1 = fo[0].rstrip('\n').split(' ')

        if date_1 != ['']:

            if len(date_1) == 6:
                date_1.remove(date_1[2])

            if date_1[1] == 'Sep' and date_1[2] == '7' and date_1[4] == '2016':


                d2 = date_1[1] + ' ' + date_1[2] + ' ' + date_1[4]

                t1 = time.strptime(d2, '%b %d %Y')
                t_epoch = time.mktime(t1)

                t_delta = t_max - t_epoch

                y_delta = slope * t_delta

                foll_count = float(foll_count_max) - y_delta

                foll_count = int(foll_count)

                fo[1] = str(foll_count)

            # if date_1[1] == 'Jul' and date_1[2] == '10' and date_1[4] == '2016':
            #
            #     d2 = date_1[1] + ' ' + date_1[2] + ' ' + date_1[4]
            #
            #     t1 = time.strptime(d2, '%b %d %Y')
            #     t_epoch = time.mktime(t1)
            #
            #     t_delta = t_max - t_epoch
            #
            #     y_delta = slope * t_delta
            #
            #     foll_count = float(foll_count_max) - y_delta
            #
            #     foll_count = int(foll_count)
            #
            #     fo[1] = str(foll_count)
            #
            # if date_1[1] == 'Jul' and date_1[2] == '11' and date_1[4] == '2016':
            #
            #     d2 = date_1[1] + ' ' + date_1[2] + ' ' + date_1[4]
            #
            #     t1 = time.strptime(d2, '%b %d %Y')
            #
            #     t_epoch = time.mktime(t1)
            #
            #     t_delta = t_max - t_epoch
            #
            #     y_delta = slope * t_delta
            #
            #     foll_count = float(foll_count_max) - y_delta
            #
            #     foll_count = int(foll_count)
            #
            #     fo[1] = str(foll_count)


    f = open('../followers/temp/follcount_'+u+'.csv','w')

    for fo in follcount_old:
        f.write(','.join(fo)+'\n')

    f.close()


#-----------------------------------

#################
# remove duplicate dates from follcount files
#################

# lines = open('../user_list/user_MASTER.csv','r').readlines()
#
# users = []
#
# for line in lines:
#     spline = line.rstrip('\n').split(',')
#     users.append(spline[0])
#
# print (len(users))
#
#
# for u in users:
#
#     lines = open('../followers/follcount/follcount_'+u+'.csv','r').readlines()
#
#     dates = []
#     dates_and_follcount = []
#
#     for line in lines:
#         spline = line.rstrip('\n').split(',')
#
#         date_s = spline[0].split(' ')
#
#         if len(date_s) == 6:
#             date_s.remove(date_s[2])
#
#         date_s = [date_s[0],date_s[1],date_s[2],date_s[4]]
#
#         if date_s not in dates:
#
#             dates.append(date_s)
#             dates_and_follcount.append(spline)
#
#         else:
#             print ("already exists")
#             print (date_s)
#
#
#     f = open('../followers/follcount/follcount_'+u+'.csv','w')
#
#     for df in dates_and_follcount:
#         f.write(','.join(df)+'\n')
#
#     f.close()


#------------------------------------

###################
# compare two follcount files and insert one specified missing dates from one file into the other
# i.e. between server follcount file and the follcount file on my computer
###################

# lines = open('../user_list/user_MASTER.csv','r').readlines()
#
# users = []
#
# for line in lines:
#     spline = line.rstrip('\n').split(',')
#     users.append(spline[0])
#
# for u in users:
#
#     lines1 = open('/Users/yi-linghwong/Desktop/follcount/follcount_'+u+'.csv','r').readlines()
#     lines2 = open('../followers/follcount/follcount_' + u + '.csv', 'r').readlines()
#
#     missing_date= []
#     date_2 = []
#
#
#     for line in lines1:
#
#         spline1 = line.rstrip('\n').split(',')
#
#         date_s = spline1[0].split(' ')
#
#         if len(date_s) == 6:
#             date_s.remove(date_s[2])
#
#         if date_s[1] == 'Jul' and date_s[2] == '19' and date_s[4] == '2016':
#
#             missing_date = spline1
#
#     dates_and_follcount = []
#
#     for line in lines2:
#
#         spline2 = line.rstrip('\n').split(',')
#         dates_and_follcount.append(spline2)
#
#         date_s = spline2[0].split(' ')
#
#         if len(date_s) == 6:
#             date_s.remove(date_s[2])
#
#         date_2.append([date_s[1],date_s[2],date_s[4]])
#
#
#     # only do the following for files that do not have Jul 19 2016
#     if ['Jul', '19', '2016'] not in date_2:
#
#         print ("no 19")
#
#         index_19 = None
#
#         for index,d2 in enumerate(date_2):
#
#             if d2[0] == 'Jul' and d2[1] == '18' and d2[2] == '2016':
#
#                 index_19 = index+1
#
#         print (index_19)
#
#         dates_and_follcount.insert(index_19,missing_date)
#
#         f = open('../followers/follcount/follcount_' + u + '.csv', 'w')
#
#         for df in dates_and_follcount:
#             f.write(','.join(df)+'\n')
#
#         f.close()


#-------------------------------------------------

###############
# test mongoDB
###############
#
# from pymongo import MongoClient
# from datetime import datetime
#
# client = MongoClient()
#
# db = client.twitter_db
#
# tweets_iterator = db.twitter_collection.find()
#
# for tweet in tweets_iterator:
#
#     #print (tweet['text'])
#     print (tweet)







