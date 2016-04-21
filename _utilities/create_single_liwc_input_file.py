__author__ = 'yi-linghwong'

import os
import sys

path_to_labelled_raw_file = '../output/engrate/labelled_space_raw.csv'
path_to_store_single_liwc_input_file = '../output/liwc/single_input/space.txt'

lines = open(path_to_labelled_raw_file,'r').readlines()

posts = []

for line in lines:

    spline = line.replace('\n','').split(',')
    posts.append(spline[0])


f = open(path_to_store_single_liwc_input_file,'w')

f.write(' '.join(posts))

f.close()