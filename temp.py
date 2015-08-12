__author__ = 'yi-linghwong'

import sys
from collections import Counter
import itertools
from nltk.util import ngrams


lines = open('output/output_engrate_label_080815_noART.csv', 'r').readlines()

l1=[]
l2=[]

for line in lines:
    spline=line.replace("\n", "").split(",")
    #creates a list with key and value. Split splits a string at the comma and stores the result in a list

    n=4


    if (spline[1] == 'HRT'):
        for i in range(1,n):
            n_grams = ngrams(spline[0].split(), i)
            gramify = [' '.join(x) for x in n_grams]
            l1.append(gramify)
            i=i+1

        #words=spline[0].split()
        #l1.append(words)


    elif spline[1] == 'LRT':
        for i in range(1,n):
            n_grams = ngrams(spline[0].split(), i)
            gramify = [' '.join(x) for x in n_grams]
            l2.append(gramify)
            i=i+1




##combine lists in a list into one long list. Flattening a list.
hrt_t = list(itertools.chain(*l1))
lrt_t = list(itertools.chain(*l2))

hrt = [w.lower() for w in hrt_t]
lrt = [w.lower() for w in lrt_t]



lines2 = open('output/extraTree_feature_importance.csv', 'r').readlines()

features=[]
for line in lines2:
    spline=line.replace("\n", "").split("\t")

    features.append(spline[1])

file = open('output/extraTree_feat_byClass.csv', 'w')

feat_by_class=[]
for f in features:
    hrt_count = hrt.count(f)
    lrt_count = lrt.count(f)

    #print ("HRT %s: " % f + str(hrt_count))
    #print ("LRT %s: " % f + str(lrt_count))

    if (hrt_count-lrt_count)>20:

        feat_by_class.append('HRT'+','+f+','+str(hrt_count))
        #file.write('HRT'+','+f+','+str(hrt_count)+'\n')

    elif (lrt_count-hrt_count)>20:
        feat_by_class.append('LRT'+','+f+','+str(lrt_count))
        #file.write('LRT'+','+f+','+str(lrt_count)+'\n')
    else:
        feat_by_class.append('BOTH'+','+f+','+str(hrt_count)+','+str(lrt_count))
        #file.write('BOTH'+','+f+','+str(hrt_count)+','+str(lrt_count)+'\n')

feat_by_class = sorted(feat_by_class)

for f in feat_by_class:
    file.write(f+'\n')





'''
from collections import Counter
list1=['apple','egg','apple','banana','egg','apple']
counts = Counter(list1)
print(counts)
Counter({'apple': 3, 'egg': 2, 'banana': 1})
'''


