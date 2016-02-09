__author__ = 'yi-linghwong'

import time
import sys
import os
import numpy as np

lines = open('updated_followers.txt','r').readlines()

eng_rate_new = []

for line in lines:
    spline = line.replace('\n','').split(',')

    if len(spline) == 8:

        engrate = str((np.divide(int(spline[4]),int(spline[2])))*100)

        eng_rate_new.append([spline[0],spline[1],spline[2],spline[3],spline[4],spline[5],engrate,spline[6],spline[7]])

    else:
        print(spline)
        pass

print ("Length of tweet list is "+str(len(eng_rate_new)))

f = open('updated_engrate.txt','w')

for ern in eng_rate_new:

    f.write(','.join(ern)+'\n')

f.close()

list = []

for ern in eng_rate_new:

    if float(ern[6]) < 0.0009:

        list.append([ern[0],ern[1],ern[6]])


print ("Length of list is "+str(len(list)))

f = open('temp2.txt','w')

for l in list:

    f.write('\t'.join(l)+'\n')

f.close()