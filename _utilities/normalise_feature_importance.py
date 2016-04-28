__author__ = 'yi-linghwong'

import sys
import os
import numpy as np
from scipy.stats import linregress
import warnings

class NormaliseFeatureImportance():

    def create_her_ler_list(self):

    #################
    # create two separate lists of all her and ler features from all fields
    #################

        lines1 = open(path_to_space_feature_score_file,'r').readlines()
        lines2 = open(path_to_politics_feature_score_file,'r').readlines()
        lines3 = open(path_to_business_feature_score_file,'r').readlines()
        lines4 = open(path_to_nonprofit_feature_score_file,'r').readlines()

        lines = lines1+lines2+lines3+lines4
        print (len(lines))

        her = []
        ler = []

        for line in lines:
            spline = line.replace('\n','').split(',')

            if spline[0] == 'HER':
                her.append(abs(float(spline[2])))

            elif spline[0] == 'LER':
                ler.append(abs(float(spline[2])))

            else:
                #print ("error")
                pass

        return her,ler


    def get_normalisation_slope(self,featimp_list):

    #################
    # calculate the normalisation factor
    #################

        x = [min(featimp_list),max(featimp_list)]
        y = [0,1]

        #her_diff = max(her) - min(her)
        #ler_diff = max(ler) - min(ler)
        #gradient_her = 1/her_diff
        #print (gradient_her)

        #################
        # linregress method returns (slope, interception, etc)
        # first item in the list returned is the slope of the linear line
        # second item in the list is the interception, c
        #################

        warnings.filterwarnings("ignore")
        slope = linregress(x, y)

        # gradient of slope

        m = slope[0]

        # intercept

        c = slope[1]


        return m,c


    def split_feature_file_by_empty_line(self,path):

        lines = open(path,'r').readlines()

        lines1 = []
        lines2 = []
        lines3 = []

        i = 0

        for line in lines:
            if line in ['\n', '\r\n']:
                i += 1
                continue

            if i == 1:
                lines1.append(line)

            if i == 2:
                lines2.append(line)

            if i == 3:
                lines3.append(line)

        return lines1,lines2,lines3


    def compare_science_and_others(self):

    #################
    # compare features in science if they exist in other fields
    # if so normalise and add to list, otherwise append 0
    #################

        her = self.create_her_ler_list()[0]

        print ("Length of science HER list is "+str(len(her)))

        # get gradient
        m = self.get_normalisation_slope(her)[0]

        # get intercept
        c = self.get_normalisation_slope(her)[1]

        ##############
        # create normalised space HER feat imp file
        ##############

        lines1 = open(path_to_space_nb_file,'r').readlines()
        lines2 = open(path_to_space_sgd_file,'r').readlines()
        lines3 = open(path_to_space_extratree_file,'r').readlines()

        lines_all = lines1+lines2+lines3

        space_her_features = []

        for line in lines_all:
            spline = line.replace('\n','').split(',')

            if spline[0] == 'HER':

                if spline[1] not in space_her_features:

                    space_her_features.append(spline[1])

        lines1 = self.split_feature_file_by_empty_line(path_to_space_feature_score_file)[0]
        lines2 = self.split_feature_file_by_empty_line(path_to_space_feature_score_file)[1]
        lines3 = self.split_feature_file_by_empty_line(path_to_space_feature_score_file)[2]

        if lines1 == [] or lines2 == [] or lines3 == []:
            print ("empty list error, exiting...")
            sys.exit()

        space_her = []

        for i in range(3):

            if i == 0:

                space_feat = []

                for line in lines1:

                    spline = line.replace('\n','').split(',')
                    space_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            space_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in space_feat:
                        space_her.append(['HER',sh,0])

            if i == 1:

                space_feat = []

                for line in lines2:

                    spline = line.replace('\n','').split(',')
                    space_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            space_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in space_feat:
                        space_her.append(['HER',sh,0])

            if i == 2:

                space_feat = []

                for line in lines3:

                    spline = line.replace('\n','').split(',')
                    space_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            space_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in space_feat:
                        space_her.append(['HER',sh,0])

        space_her.sort(key=lambda x: x[2], reverse=True)

        space_her_sorted = []

        for sh in space_her:
            sh[2] = str(sh[2])
            space_her_sorted.append(sh)
        

        f = open(path_to_store_normalised_space_feature_file,'w')

        for sh in space_her:
            f.write(','.join(sh)+'\n')

        f.close()

        ##############
        # create normalised politics HER feat imp file
        ##############

        lines1 = self.split_feature_file_by_empty_line(path_to_politics_feature_score_file)[0]
        lines2 = self.split_feature_file_by_empty_line(path_to_politics_feature_score_file)[1]
        lines3 = self.split_feature_file_by_empty_line(path_to_politics_feature_score_file)[2]

        if lines1 == [] or lines2 == [] or lines3 == []:
            print ("empty list error, exiting...")
            sys.exit()

        politics_her = []

        for i in range(3):

            if i == 0:

                politics_feat = []

                for line in lines1:

                    spline = line.replace('\n','').split(',')
                    politics_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            politics_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            politics_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            politics_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in politics_feat:
                        politics_her.append(['HER',sh,0])


            if i == 1:

                politics_feat = []

                for line in lines2:

                    spline = line.replace('\n','').split(',')
                    politics_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            politics_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            politics_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            politics_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in politics_feat:
                        politics_her.append(['HER',sh,0])

            if i == 2:

                politics_feat = []

                for line in lines3:

                    spline = line.replace('\n','').split(',')
                    politics_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            politics_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            politics_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            politics_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in politics_feat:
                        politics_her.append(['HER',sh,0])


        politics_her.sort(key=lambda x: x[2], reverse=True)

        politics_her_sorted = []

        for ph in politics_her:
            ph[2] = str(ph[2])
            politics_her_sorted.append(ph)

        f = open(path_to_store_normalised_politics_feature_file,'w')

        for ph in politics_her_sorted:
            f.write(','.join(ph)+'\n')

        f.close()

        ##############
        # create normalised business HER feat imp file
        ##############

        lines1 = self.split_feature_file_by_empty_line(path_to_business_feature_score_file)[0]
        lines2 = self.split_feature_file_by_empty_line(path_to_business_feature_score_file)[1]
        lines3 = self.split_feature_file_by_empty_line(path_to_business_feature_score_file)[2]

        if lines1 == [] or lines2 == [] or lines3 == []:
            print ("empty list error, exiting...")
            sys.exit()

        business_her = []

        for i in range(3):

            if i == 0:

                business_feat = []

                for line in lines1:

                    spline = line.replace('\n','').split(',')
                    business_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            business_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            business_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            business_her.append(['HER',spline[1],featimp_norm])


                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in business_feat:
                        business_her.append(['HER',sh,0])

            if i == 1:

                business_feat = []

                for line in lines2:

                    spline = line.replace('\n','').split(',')
                    business_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            business_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            business_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            business_her.append(['HER',spline[1],featimp_norm])


                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in business_feat:
                        business_her.append(['HER',sh,0])

            if i == 2:

                business_feat = []

                for line in lines3:

                    spline = line.replace('\n','').split(',')
                    business_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            business_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            business_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            business_her.append(['HER',spline[1],featimp_norm])


                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in business_feat:
                        business_her.append(['HER',sh,0])


        business_her.sort(key=lambda x: x[2], reverse=True)

        business_her_sorted = []

        for bh in business_her:
            bh[2] = str(bh[2])
            business_her_sorted.append(bh)

        f = open(path_to_store_normalised_business_feature_file,'w')

        for bh in business_her_sorted:
            f.write(','.join(bh)+'\n')

        f.close()

        ##############
        # create normalised nonprofit HER feat imp file
        ##############

        lines1 = self.split_feature_file_by_empty_line(path_to_nonprofit_feature_score_file)[0]
        lines2 = self.split_feature_file_by_empty_line(path_to_nonprofit_feature_score_file)[1]
        lines3 = self.split_feature_file_by_empty_line(path_to_nonprofit_feature_score_file)[2]

        if lines1 == [] or lines2 == [] or lines3 == []:
            print ("empty list error, exiting...")
            sys.exit()

        nonprofit_her = []

        for i in range(3):

            if i == 0:

                nonprofit_feat = []

                for line in lines1:

                    spline = line.replace('\n','').split(',')
                    nonprofit_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            nonprofit_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            nonprofit_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            nonprofit_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in nonprofit_feat:
                        nonprofit_her.append(['HER',sh,0])

            if i == 1:

                nonprofit_feat = []

                for line in lines2:

                    spline = line.replace('\n','').split(',')
                    nonprofit_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            nonprofit_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            nonprofit_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            nonprofit_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in nonprofit_feat:
                        nonprofit_her.append(['HER',sh,0])

            if i == 2:

                nonprofit_feat = []

                for line in lines3:

                    spline = line.replace('\n','').split(',')
                    nonprofit_feat.append(spline[1])

                    if spline[1] in space_her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            nonprofit_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            nonprofit_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            nonprofit_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_her_features:
                    if sh not in nonprofit_feat:
                        nonprofit_her.append(['HER',sh,0])

        nonprofit_her.sort(key=lambda x: x[2], reverse=True)

        nonprofit_her_sorted = []

        for nh in nonprofit_her:
            nh[2] = str(nh[2])
            nonprofit_her_sorted.append(nh)

        f = open(path_to_store_normalised_nonprofit_feature_file,'w')

        for nh in nonprofit_her_sorted:
            f.write(','.join(nh)+'\n')

        f.close()

        #-------------------------------------------------------------
        # LER

        ler = self.create_her_ler_list()[1]

        print ("Length of science LER list is "+str(len(ler)))

        # get gradient
        m2 = self.get_normalisation_slope(ler)[0]

        # get intercept
        c2 = self.get_normalisation_slope(ler)[1]

        ##############
        # create normalised space HER feat imp file
        ##############

        lines1 = open(path_to_space_nb_file,'r').readlines()
        lines2 = open(path_to_space_sgd_file,'r').readlines()
        lines3 = open(path_to_space_extratree_file,'r').readlines()

        lines_all = lines1+lines2+lines3

        space_ler_features = []

        for line in lines_all:
            spline = line.replace('\n','').split(',')

            if spline[0] == 'LER':

                if spline[1] not in space_ler_features:

                    space_ler_features.append(spline[1])

        lines1 = self.split_feature_file_by_empty_line(path_to_space_feature_score_file)[0]
        lines2 = self.split_feature_file_by_empty_line(path_to_space_feature_score_file)[1]
        lines3 = self.split_feature_file_by_empty_line(path_to_space_feature_score_file)[2]

        if lines1 == [] or lines2 == [] or lines3 == []:
            print ("empty list error, exiting...")
            sys.exit()

        space_ler = []

        for i in range(3):

            if i == 0:

                space_feat = []

                for line in lines1:

                    spline = line.replace('\n','').split(',')
                    space_feat.append(spline[1])

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            space_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in space_feat:
                        space_ler.append(['LER',sh,0])

            if i == 1:

                space_feat = []

                for line in lines2:

                    spline = line.replace('\n','').split(',')
                    space_feat.append(spline[1])

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            space_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in space_feat:
                        space_ler.append(['LER',sh,0])

            if i == 2:

                space_feat = []

                for line in lines3:

                    spline = line.replace('\n','').split(',')
                    space_feat.append(spline[1])

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            space_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            space_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in space_feat:
                        space_ler.append(['LER',sh,0])

        space_ler.sort(key=lambda x: x[2], reverse=True)

        space_ler_sorted = []

        for sh in space_ler:
            sh[2] = str(sh[2])
            space_ler_sorted.append(sh)


        f = open(path_to_store_normalised_space_feature_file,'a')

        for sh in space_ler:
            f.write(','.join(sh)+'\n')

        f.close()

        ##############
        # create normalised politics LER feat imp file
        ##############

        lines1 = self.split_feature_file_by_empty_line(path_to_politics_feature_score_file)[0]
        lines2 = self.split_feature_file_by_empty_line(path_to_politics_feature_score_file)[1]
        lines3 = self.split_feature_file_by_empty_line(path_to_politics_feature_score_file)[2]

        if lines1 == [] or lines2 == [] or lines3 == []:
            print ("empty list error, exiting...")
            sys.exit()

        politics_ler = []

        for i in range(3):

            if i == 0:

                politics_feat = []

                for line in lines1:

                    spline = line.replace('\n','').split(',')

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            politics_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a LER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            politics_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for LER features which are not in space's LER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            politics_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in politics_feat:
                        politics_ler.append(['LER',sh,0])

            if i == 1:

                politics_feat = []

                for line in lines2:

                    spline = line.replace('\n','').split(',')

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            politics_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a LER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            politics_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for LER features which are not in space's LER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            politics_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in politics_feat:
                        politics_ler.append(['LER',sh,0])

            if i == 2:

                politics_feat = []

                for line in lines3:

                    spline = line.replace('\n','').split(',')

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            politics_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a LER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            politics_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for LER features which are not in space's LER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            politics_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in politics_feat:
                        politics_ler.append(['LER',sh,0])

        politics_ler.sort(key=lambda x: x[2], reverse=True)

        politics_ler_sorted = []

        for pl in politics_ler:
            pl[2] = str(pl[2])
            politics_ler_sorted.append(pl)

        f = open(path_to_store_normalised_politics_feature_file,'a')

        for pl in politics_ler_sorted:
            f.write(','.join(pl)+'\n')

        f.close()

        ##############
        # create normalised business LER feat imp file
        ##############

        lines1 = self.split_feature_file_by_empty_line(path_to_business_feature_score_file)[0]
        lines2 = self.split_feature_file_by_empty_line(path_to_business_feature_score_file)[1]
        lines3 = self.split_feature_file_by_empty_line(path_to_business_feature_score_file)[2]

        if lines1 == [] or lines2 == [] or lines3 == []:
            print ("empty list error, exiting...")
            sys.exit()

        business_ler = []

        for i in range(3):

            if i == 0:

                business_feat = []

                for line in lines1:

                    spline = line.replace('\n','').split(',')

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            business_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a LER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            business_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for LER features which are not in space's LER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            business_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in business_feat:
                        business_ler.append(['LER',sh,0])

            if i == 1:

                business_feat = []

                for line in lines2:

                    spline = line.replace('\n','').split(',')

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            business_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a LER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            business_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for LER features which are not in space's LER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            business_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in business_feat:
                        business_ler.append(['LER',sh,0])

            if i == 2:

                business_feat = []

                for line in lines3:

                    spline = line.replace('\n','').split(',')

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            business_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a LER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            business_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for LER features which are not in space's LER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            business_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in business_feat:
                        business_ler.append(['LER',sh,0])

        business_ler.sort(key=lambda x: x[2], reverse=True)

        business_ler_sorted = []

        for bl in business_ler:
            bl[2] = str(bl[2])
            business_ler_sorted.append(bl)

        f = open(path_to_store_normalised_business_feature_file,'a')

        for bl in business_ler_sorted:
            f.write(','.join(bl)+'\n')

        f.close()

        ##############
        # create normalised nonprofit LER feat imp file
        ##############

        lines1 = self.split_feature_file_by_empty_line(path_to_nonprofit_feature_score_file)[0]
        lines2 = self.split_feature_file_by_empty_line(path_to_nonprofit_feature_score_file)[1]
        lines3 = self.split_feature_file_by_empty_line(path_to_nonprofit_feature_score_file)[2]

        if lines1 == [] or lines2 == [] or lines3 == []:
            print ("empty list error, exiting...")
            sys.exit()

        nonprofit_ler = []

        for i in range(3):

            if i == 0:

                nonprofit_feat = []

                for line in lines1:

                    spline = line.replace('\n','').split(',')

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            nonprofit_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            nonprofit_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for LER features which are not in space's LER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            nonprofit_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in nonprofit_feat:
                        nonprofit_ler.append(['LER',sh,0])

            if i == 1:

                nonprofit_feat = []

                for line in lines2:

                    spline = line.replace('\n','').split(',')

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            nonprofit_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            nonprofit_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for LER features which are not in space's LER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            nonprofit_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in nonprofit_feat:
                        nonprofit_ler.append(['LER',sh,0])

            if i == 2:

                nonprofit_feat = []

                for line in lines3:

                    spline = line.replace('\n','').split(',')

                    if spline[1] in space_ler_features:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            nonprofit_ler.append(['LER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'HER':
                            nonprofit_ler.append(['LER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for LER features which are not in space's LER list, just normalise and append as is
                    else:

                        if spline[0] == 'LER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m2 * (featimp_ori)) + c2),4)
                            nonprofit_ler.append(['LER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for sh in space_ler_features:
                    if sh not in nonprofit_feat:
                        nonprofit_ler.append(['LER',sh,0])

        nonprofit_ler.sort(key=lambda x: x[2], reverse=True)

        nonprofit_ler_sorted = []

        for nl in nonprofit_ler:
            nl[2] = str(nl[2])
            nonprofit_ler_sorted.append(nl)

        f = open(path_to_store_normalised_nonprofit_feature_file,'a')

        for nl in nonprofit_ler_sorted:
            f.write(','.join(nl)+'\n')

        f.close()


################
# variables
################

path_to_space_feature_score_file = '../output/featimp_normalisation/extratree/space.csv'
path_to_politics_feature_score_file = '../output/featimp_normalisation/extratree/politics.csv'
path_to_business_feature_score_file = '../output/featimp_normalisation/extratree/business.csv'
path_to_nonprofit_feature_score_file = '../output/featimp_normalisation/extratree/nonprofit.csv'

path_to_store_normalised_space_feature_file = '../output/featimp_normalisation/extratree/normalised_space.csv'
path_to_store_normalised_politics_feature_file = '../output/featimp_normalisation/extratree/normalised_politics.csv'
path_to_store_normalised_business_feature_file = '../output/featimp_normalisation/extratree/normalised_business.csv'
path_to_store_normalised_nonprofit_feature_file = '../output/featimp_normalisation/extratree/normalised_nonprofit.csv'

# DO NOT change the following!
path_to_space_nb_file = '../output/featimp_normalisation/nb/space.csv'
path_to_space_sgd_file = '../output/featimp_normalisation/sgd/space.csv'
path_to_space_extratree_file = '../output/featimp_normalisation/extratree/space.csv'


if __name__ == '__main__':

    nf = NormaliseFeatureImportance()

    #nf.create_her_ler_list()
    #nf.get_normalisation_slope()
    nf.compare_science_and_others()

