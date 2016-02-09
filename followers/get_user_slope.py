__author__ = 'yi-linghwong'

#############
# Get the slope of the linear equation for a list of user
# y = mx + c, where y is number of follower, x is epoch time
#############

import sys
import os
from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import interp1d
from decimal import Decimal
import time


#lines = open('user_list_old.txt','r').readlines()

user_list = ['MarsCuriosity']

'''
for line in lines:
    spline = line.replace('\n','')
    user_list.append(spline)
'''


user_slope_list = []

for ul in user_list:

    user_slope = []

    lines = open('follcount_'+ul+'.csv','r').readlines()
    #lines = open('test.txt','r').readlines()

    follcount_list = []
    date = []


    for line in lines:
        spline = line.replace('\n','').split(',')

        follcount_list.append(float(spline[1]))
        date.append(spline[0])

    date_split = []

    for d in date:
        d1 = d.replace('\n','').split(' ')

        if len(d1) == 6:
            d1.remove(d1[2])

        date_split.append(d1)


    date_epoch = []

    for ds in date_split:

        date_s = ds[1]+' '+ds[2]+' '+ds[4]


        t1 = time.strptime(date_s,'%b %d %Y')
        t_epoch = time.mktime(t1)
        date_epoch.append(t_epoch)


    x = date_epoch
    y = follcount_list

    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(x)

    plt.plot(x, y, '*')
    plt.plot(x, ys)


    plt.show()

    user_slope.append(ul)

    # linregress method returns (slope, interception, etc)
    # first item in the list returned is the slope of the linear line

    user_slope.append(str(linregress(x, y)[0]))

    user_slope_list.append(user_slope)

'''
# write results to a file
f = open('user_slope.txt','w')

for usl in user_slope_list:
    f.write(','.join(usl)+'\n')

f.close()

'''













