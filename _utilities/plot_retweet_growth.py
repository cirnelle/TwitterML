__author__ = 'yi-linghwong'

import sys
import os
import time
import matplotlib.pyplot as plt
import pylab
import numpy as np
from scipy.stats import linregress
from statsmodels.robust.scale import mad
import statistics

lines = open('../tweets/raw_space.csv','r').readlines()


user_list= []

for line in lines[1:]:
    spline = line.replace('\n', '').split(',')

    if spline[0] not in user_list:
        user_list.append(spline[0])

# rt_count = []
# foll_count = []
#
# for ul in user_list:
#
#     rt = []
#
#     for line in lines:
#         spline = line.replace('\n', '').split(',')
#
#         if spline[0] == ul:
#
#             rt.append(int(spline[5]))
#
#             if len(rt) > 1000:
#                 average = np.mean(rt)
#                 rt_count.append(average)
#                 foll_count.append(int(spline[3]))
#                 break
#
# ################
# # remove outliars
# ################
#
# rt_mad = mad(foll_count)
# rt_median = np.median(foll_count)
# rt_top_thresh = round((rt_median + (1e-10 * rt_mad)), 2)  # 5 is just an arbitrary number we choose
#
# print("MAD for rt is " + str(rt_mad))
# print("Median for rt is " + str(rt_median))
# print("Top threshold for rt is " + str(rt_top_thresh))
#
# foll_outliers = []
# foll_count_ = []
# index = []
#
# for i, rc in enumerate(foll_count):
#
#     if rc <= rt_top_thresh:
#         foll_count_.append(rc)
#         index.append(i)
#
#     else:
#         foll_outliers.append(rc)
#
# print ("length of outliar list is "+str(len(foll_outliers)))
#
# rt_count_ = []
#
# for j, de in enumerate(rt_count):
#     if j in index:
#         rt_count_.append(de)
#
#
#
#
# x = foll_count_
# y = rt_count_
#
# # mean_retweet = np.mean(y)
#
# coefficients = np.polyfit(x, y, 1)
# polynomial = np.poly1d(coefficients)
# ys = polynomial(x)
#
# # get standard deviation of follower count
#
#
# # error = abs(ys - y)
# # mean_error = error.mean()
# # error_std = (ys - y).std()
# # error_std_percent = round(((error_std / mean_retweet) * 100), 2)
#
# # print ("Mean follcount for "+str(ul)+" is "+str(mean_follcount)+". Mean error is "+str(mean_error)+". Standard deviation is "+str(error_std)+"("+str(error_std_percent)+"%)")
#
# plt.plot(x, y, '*', label=str('retweet'))
# plt.xlabel('RT')
# plt.ylabel('foll')
#
# # plt.errorbar(x, ys, error, label='std error ' + str(error_std_percent) + '%')
# # pylab.legend(loc='upper left')
# plt.plot(x, ys)
#
# plt.show()
#
# slope = linregress(x, y)[0]
#
# print (slope)
#
# sys.exit()




for ul in user_list:

    retweet_counts = []
    date = []

    for line in lines:
        spline = line.replace('\n','').split(',')

        if spline[0] == ul:

            retweet_counts.append(int(spline[5]))
            date.append(spline[1])

    date_split = []

    for d in date:
        d1 = d.replace('\n', '').split(' ')

        if len(d1) == 7:
            d1.remove(d1[2])

        date_split.append(d1)

    print (date_split[:5])

    date_epoch = []

    for ds in date_split:

        if len(ds) == 6:
            date_s = ds[1] + ' ' + ds[2] + ' ' + ds[5]

            t1 = time.strptime(date_s, '%b %d %Y')
            t_epoch = time.mktime(t1)
            date_epoch.append(t_epoch)

        else:
            print ("error")


    ################
    # get foll/RT
    ################


    rt_avg = []

    for line in lines:
        spline = line.replace('\n','').split(',')

        if spline[0] == ul:

            rt_avg.append(int(spline[5]))

            if (len(rt_avg)) > 0:
                average = np.mean(rt_avg)
                corr_factor = int(spline[3]) / average
                break


    ################
    # remove outliars
    ################

    rt_mad = mad(retweet_counts)
    rt_median = np.median(retweet_counts)
    rt_top_thresh = round((rt_median + (50 * rt_mad)), 2)  # 5 is just an arbitrary number we choose

    print("MAD for rt is " + str(rt_mad))
    print("Median for rt is " + str(rt_median))
    print("Top threshold for rt is " + str(rt_top_thresh))

    rt_outliers = []
    retweets = []
    index = []

    for i, rc in enumerate(retweet_counts):

        if rc <= rt_top_thresh:
            retweets.append(rc)
            index.append(i)

        else:
            rt_outliers.append(rc)

    date = []

    for j,de in enumerate(date_epoch):
        if j in index:
           date.append(de)


    #############
    # plot graph
    #############

    x = date
    y = retweets

    #mean_retweet = np.mean(y)

    coefficients = np.polyfit(x, y, 3)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(x)

    # get standard deviation of follower count


    # error = abs(ys - y)
    # mean_error = error.mean()
    # error_std = (ys - y).std()
    # error_std_percent = round(((error_std / mean_retweet) * 100), 2)

    # print ("Mean follcount for "+str(ul)+" is "+str(mean_follcount)+". Mean error is "+str(mean_error)+". Standard deviation is "+str(error_std)+"("+str(error_std_percent)+"%)")

    plt.plot(x, y, '*', label=str('retweet'))
    plt.xlabel('Epoch time')
    plt.ylabel('retweet count')

    #plt.errorbar(x, ys, error, label='std error ' + str(error_std_percent) + '%')
    #pylab.legend(loc='upper left')
    plt.plot(x, ys)

    plt.show()


    # linregress method returns (slope, interception, etc)
    # first item in the list returned is the slope of the linear line

    slope = linregress(x, y)[0]
    foll_slope = slope * corr_factor
    print (slope)
    print (foll_slope)

    sys.exit()





