__author__ = 'yi-linghwong'

import time
import sys
import os
import numpy as np

lines = open('updated_followers.txt','r').readlines()

eng_rate_new = []

for line in lines:
    spline = line.replace('\n','').split(',')

    engrate = (np.divide(spline[4],spline[2]))*100

    eng_rate_new.append([spline[0],spline[1],spline[2],spline[3],spline[4],spline[5],engrate,spline[6],spline[7]])

print (len(eng_rate_new))