__author__ = 'yi-linghwong'

import os
import sys
import numpy as np
import statistics


class GetLiwcMean():

    def get_mean_and_std(self):

        lines = open(path_to_liwc_result_file,'r').readlines()

        liwc_cat_name = []

        for line in lines[:1]:
            spline = line.replace('\n','').split('\t')

            for index,s in enumerate(spline):

                if s == 'Analytic':
                    analytic_index = index

            liwc_cat_name.extend(spline[analytic_index:])

        print (liwc_cat_name)

        liwc = []
        liwc_score = []
        liwc_scores = []

        #analytic,clout,authentic,tone,wps,sixltr,posemo,negemo,anx,ang,sad,insight,cause,discrep,tentat,certain,differ,see,hear,feel,affiliation,achieve,power,reward,risk,swear,netspeak,assent,nonflu,filler,qmark,exclam = []


        for line in lines[1:]:

            spline = line.replace('\n','').split('\t')
            liwc.append(spline[analytic_index:])


        # convert string into float
        for l in liwc:

            liwc_score = [float(i) for i in l]
            liwc_scores.append(liwc_score)

        liwc_m = np.mean(liwc_scores, axis = 0)
        liwc_s = np.std(liwc_scores, axis = 0)


        # convert float to string

        liwc_mean = []
        liwc_std = []

        for lm in liwc_m:
            lm = str(round((lm),2))
            liwc_mean.append(lm)

        for ls in liwc_s:
            ls = str(round((ls),2))
            liwc_std.append(ls)

        print ("Length of cat name list is "+str(len(liwc_cat_name)))
        print ("Length of mean list is "+str(len(liwc_mean)))
        print ("Length of std list is "+str(len(liwc_std)))


        if len(liwc_cat_name) == len(liwc_mean) == len(liwc_std):

            zipped = zip(liwc_cat_name,liwc_mean,liwc_std)

            liwc_all = []

            for z in zipped:
                z = list(z)
                liwc_all.append(z)

        else:
            print ("Length not equal, exiting...")
            sys.exit()

        # add header
        header = ['category','mean','std']
        liwc_all.insert(0,header)


        # write to file
        f = open(path_to_store_liwc_mean_and_std_file,'w')

        for la in liwc_all:
            f.write(','.join(la)+'\n')

        f.close()


################
# variables
################

path_to_liwc_result_file = '../output/liwc/sydscifest/liwc_sydscifest_ler.txt'
path_to_store_liwc_mean_and_std_file = '../output/sydscifest/sydscifest_ler_mean.txt'


if __name__ == '__main__':

    lm = GetLiwcMean()

    lm.get_mean_and_std()

