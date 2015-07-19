__author__ = 'yi-linghwong'


import os
import csv

if os.path.isfile('users_18July2015.txt'):
    lines = open('users_18July2015.txt','r').readlines()


else:
    print ("Path not found")
    sys.exit(1)

user_list=[]

for line in lines:
    spline=line.replace("\n", "").split(",")
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    user_list.append(spline[0])

print (user_list)






