__author__ = 'yi-linghwong'

lines = open('output/output_feature_importance.txt', 'r').readlines()
lines2 = open('output/output_vocab.txt', 'r').readlines()

for i in range(len(lines)):

    if float(lines[i].replace('\n',''))>0.001:

        print (lines[i].replace('\n',''),lines2[i].replace('\n',''))

