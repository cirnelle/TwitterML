__author__ = 'yi-linghwong'

import sys
import os
import numpy as np

lines1 = open('output/output_engrate_150815.csv', 'r').readlines()
lines2 = open('user_science.csv', 'r').readlines()


l1=[]

for line in lines2:
    spline2 = line.replace('\n','').split(',')

    for line in lines1:
        spline1 = line.replace('\n','').split(',')

        if (spline1[0] == spline2[0]):
            l1.append(line)


f = open('output/output_engrate_science.csv', 'w')

for l in l1:
    f.write(str(l))

f.close()





