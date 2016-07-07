__author__ = 'yi-linghwong'

import os
import sys

lines = open('/Users/yi-linghwong/Documents/PhD/sydney_science_festival/twitter/maasmuseum/maasmuseum_jan-mar2014.csv', 'r').readlines()
print (len(lines))

for line in lines[:1]:
    spline = line.rstrip('\n').replace('\r', '').replace('\t',' ').split(',')
    length = len(spline)

tweets = []

for line in lines[1:]:
    print (repr(line))
    spline = line.rstrip('\n').replace('\r', '').replace('\t', ' ').split(',')

    if len(spline) < length:

        print(spline)

#     if len(spline) == length:
#         tweets.append(spline)
#
#     else:
#
#         join_until = len(spline) - 37
#         text = spline[2:join_until]
#         comma_removed = ' '.join(text)
#
#         # delete the text containing commas and replace it with the no-commas text
#         del spline[2:join_until]
#         spline.insert(2,comma_removed)
#
#         if len(spline) == 40:
#             tweets.append(spline)
#
#         else:
#             print ("error")
#             print (spline)
#
#
# f = open('/Users/yi-linghwong/Documents/PhD/sydney_science_festival/twitter/maasmuseum/maasmuseum_ALL.csv','w')
#
# for t in tweets:
#     f.write(','.join(t)+'\n')
#
# f.close()
