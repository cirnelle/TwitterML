__author__ = 'yi-linghwong'

import os
import sys
import pandas as pd
from datetime import datetime


lines1 = open('../user_list/user_MASTER.csv','r').readlines()

users = []

for line in lines1:
    spline = line.rstrip('\n').split(',')
    users.append(spline[0])

print (len(users))

for u in users:

    print (u)

    #-----------------
    # find start and end date to get date range

    lines = open('../followers/follcount/follcount_'+u+'.csv','r').readlines()

    dates_ori = []

    for line in lines:
        spline = line.rstrip('\n').split(',')

        if spline != ['']:
            dates_ori.append(spline[0])

    d2_start = dates_ori[0].replace('\n', '').split(' ')
    d2_end = dates_ori[len(dates_ori) - 1].replace('\n', '').split(' ')

    if (len(d2_start)) == 6:
        d2_start.remove(d2_start[2])

    date_s = d2_start[1] + ' ' + d2_start[2] + ' ' + d2_start[4]

    t2 = datetime.strptime(date_s, '%b %d %Y').date()

    date_start = str(t2.month) + '-' + str(t2.day) + '-' + str(t2.year)

    if (len(d2_end)) == 6:
        d2_end.remove(d2_end[2])

    date_s = d2_end[1] + ' ' + d2_end[2] + ' ' + d2_end[4]

    t3 = datetime.strptime(date_s, '%b %d %Y').date()

    date_end = str(t3.month) + '-' + str(t3.day) + '-' + str(t3.year)

    idx = pd.date_range(date_start,date_end)

    #################
    # use Pandas reindex to extrapolate the missing dates
    #################

    date_follcount_dict = {}

    for line in lines:

        spline = line.rstrip('\n').split(',')

        if spline != [''] and spline[1] != 'nan':
            d1 = spline[0].replace('\n','').split(' ')

            if (len(d1)) == 6:
                d1.remove(d1[2])

            date_s = d1[1] + ' ' + d1[2] + ' ' + d1[4]

            t1 = datetime.strptime(date_s,'%b %d %Y').date()

            date = str(t1.year) + '-' + str(t1.month) + '-' + str(t1.day)

            date_follcount_dict[date] = spline[1]


    #print ("Length of original list is: "+str(len(date_follcount_dict)))

    s = pd.Series(date_follcount_dict)

    s.index = pd.DatetimeIndex(s.index)

    # reindex fills in the missing dates with the fill value we define

    s = s.reindex(idx, fill_value='0')

    dates = []
    follcount = []

    for i in s.index:

        date = str(i.year)+'-'+str(i.month)+'-'+str(i.day)
        dates.append(date)

    for j in s.values:

        follcount.append(j)

    zipped = zip(dates,follcount)

    dates_follcount = []

    for z in zipped:
        z = list(z)
        dates_follcount.append(z)

    #print ("Length of extrapolated list is: "+str(len(dates_follcount)))


    ####################
    # interpolate follower count for missing dates
    ####################

    #dates_follcount = [['date1','4'],['date2','6'],['date2','0'],['date2','4'],['date2','8'],['date2','9'],['date2','0'],['date2','11'],['date2','12']]

    follcount_index_dict = {}

    follcount = []
    dates = []

    for index,df in enumerate(dates_follcount):
        follcount.append(df[1])
        dates.append(df[0])
        follcount_index_dict[index] = df[1]

    temp = []
    foll_diff = []

    for index,f in enumerate(follcount):

        if f == '0':
            temp.append(index)

            if follcount_index_dict[index-1] != '0':
                foll_diff.append(int(follcount_index_dict[index-1]))

        elif f != '0' and len(temp) > 0:

            index_diff = len(temp) + 1
            foll_diff.append(int(follcount_index_dict[index]))

            follcount_diff = foll_diff[1] - foll_diff[0]

            step = int(follcount_diff / index_diff)

            for n in range(index_diff - 1):
                target_index = temp[n]
                follcount_index_dict[target_index] = str(foll_diff[0] + (step*(n+1)))

            temp = []
            foll_diff = []


    follcount_interpolated = []

    for key,value in follcount_index_dict.items():
        follcount_interpolated.append(value)

    if len(dates) == len(follcount_interpolated):

        date_follcount_interpolated = []

        zipped = zip(dates,follcount_interpolated)

        for z in zipped:
            z = list(z)
            date_follcount_interpolated.append(z)

        f = open('../followers/follcount_interpolated/'+u+'.csv','w')

        for df in date_follcount_interpolated:
            f.write(','.join(df)+'\n')

        f.close()

    else:
        print ("Lists of different lengths, exiting...")
        sys.exit()
