__author__ = 'yi-linghwong'

##############
# remove duplicated entries with the same date from follower growth files
##############

import os
import sys

lines = open('user_list.txt','r').readlines()

users = []

for line in lines:
    spline = line.replace('\n','')
    users.append(spline)

for u in users:

    lines = open('follcount_'+str(u)+'.csv','r').readlines()

    full_line = []
    dates = []
    new_list = []

    for line in lines:
        spline = line.replace('\n','').split(' ')

        if len(spline) == 6:
            joined = ' '.join(spline[:4])

        elif len(spline) == 5:
            joined = ' '.join(spline[:3])

        new_list.append([joined,spline[-2],spline[-1]])

    #print (new_list)


    temp = []
    unique_lines = []

    for nl in new_list:

        if nl[0] not in temp:
            temp.append(nl[0])
            unique_lines.append(nl)


    final_list = []

    for ul in unique_lines:
        merged = ' '.join(ul)
        final_list.append(merged)


    f = open("follcount_"+str(u)+".csv",'w')

    for fl in final_list:
        f.write(fl+'\n')

    f.close()



