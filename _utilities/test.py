__author__ = 'yi-linghwong'

import os
import sys

lines = open('../tweets/events/sydscifest_2016/from_jim/raw_sydscifest','r').readlines()

labelled_raw = []

for line in lines[:60]:
    spline = line.replace('\n','').split(',')
    labelled_raw.append([spline[-1],'HER',spline[-2]])

print (len(labelled_raw))

for line in lines[60:]:
    spline = line.replace('\n', '').split(',')
    labelled_raw.append([spline[-1],'LER', spline[-2]])

print(len(labelled_raw))

f = open('../output/engrate/sydscifest/from_jim/labelled_raw.csv','w')

for lr in labelled_raw:
    f.write(','.join(lr)+'\n')

f.close()


######################
# preprocessed
######################

lines = open('../tweets/events/sydscifest_2016/from_jim/preprocessed_sydscifest.csv','r').readlines()

labelled= []

for line in lines[:60]:
    spline = line.replace('\n','').split(',')


    labelled.append([spline[-1],'HER'])

print (len(labelled))

for line in lines[60:]:
    spline = line.replace('\n', '').split(',')
    labelled.append([spline[-1],'LER'])

print(len(labelled))

f = open('../output/engrate/sydscifest/from_jim/labelled.csv','w')

for l in labelled:
    f.write(','.join(l)+'\n')

f.close()
