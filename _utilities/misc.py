__author__ = 'yi-linghwong'

import os
import sys

#-------------------------------
# get engrate from excel and then insert into csv (maas data)

lines1 = open('../output/engrate/maas/engrate_from_excel.csv','r').readlines()
lines2 = open('../tweets/maas/raw_maasmuseum.csv','r').readlines()

id_dict = {}

for line in lines1:
    spline = line.rstrip('\n').split(',')

    id_dict[spline[0]] = spline[1]

tweets = []

for line in lines2:
    spline = line.rstrip('\n').split(',')
    key = spline[2]

    if key in id_dict:
        engrate = id_dict[key]
        spline.insert(7,engrate)
        tweets.append(spline)

    else:
        print ("key does not exist")
        print (key)

print (len(tweets))

f = open('../output/engrate/maas/maasmuseum/engrate_maasmuseum_raw.csv','w')

for t in tweets:
    f.write(','.join(t)+'\n')

f.close()

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



