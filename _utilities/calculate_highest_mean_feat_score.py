__author__ = 'yi-linghwong'

import os
import sys
import numpy as np
from scipy.stats import linregress
import warnings

class CalculateHighestMean():

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


    def get_normalised_feat_score(self):


        her = self.create_her_ler_list()[0]

        print ("Length of HER list is "+str(len(her)))

        # get gradient
        m = self.get_normalisation_slope(her)[0]

        # get intercept
        c = self.get_normalisation_slope(her)[1]

        #############
        # get all features for a field from all three classifiers
        #############

        lines1 = open(path_to_nb_file,'r').readlines()
        lines2 = open(path_to_sgd_file,'r').readlines()
        lines3 = open(path_to_extratree_file,'r').readlines()

        lines_all = lines1+lines2+lines3

        her_features = []

        for line in lines_all:
            spline = line.replace('\n','').split(',')

            if spline[0] == 'HER':

                if spline[1] not in her_features:

                    her_features.append(spline[1])

        ##############
        # create normalised HER feat imp file
        ##############

        lines1 = self.split_feature_file_by_empty_line(path_to_feature_score_file_cf)[0]
        lines2 = self.split_feature_file_by_empty_line(path_to_feature_score_file_cf)[1]
        lines3 = self.split_feature_file_by_empty_line(path_to_feature_score_file_cf)[2]

        if lines1 == [] or lines2 == [] or lines3 == []:
            print ("empty list error, exiting...")
            sys.exit()

        field_her = []

        for i in range(3):

            if i == 0:

                field_feat = []

                for line in lines1:

                    spline = line.replace('\n','').split(',')
                    field_feat.append(spline[1])

                    if spline[1] in her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            field_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            field_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            field_her.append(['HER',spline[1],featimp_norm])

                # for features which are in a field's feature list but not in the other field's feature list
                for hf in her_features:
                    if hf not in field_feat:
                        field_her.append(['HER',hf,0])

            if i == 1:

                field_feat = []

                for line in lines2:

                    spline = line.replace('\n','').split(',')
                    field_feat.append(spline[1])

                    if spline[1] in her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            field_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            field_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            field_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for hf in her_features:
                    if hf not in field_feat:
                        field_her.append(['HER',hf,0])

            if i == 2:

                field_feat = []

                for line in lines3:

                    spline = line.replace('\n','').split(',')
                    field_feat.append(spline[1])

                    if spline[1] in her_features:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            field_her.append(['HER',spline[1],featimp_norm])

                        # if the feature is not a HER feature then append '0' as its feat importance
                        elif spline[0] == 'LER':
                            field_her.append(['HER',spline[1],0])


                        else:
                            #print ("error")
                            pass

                    # for HER features of politics which are not in space's HER list, just normalise and append as is
                    else:

                        if spline[0] == 'HER':
                            featimp_ori = abs(float(spline[2]))
                            featimp_norm = round(((m * (featimp_ori)) + c),4)
                            field_her.append(['HER',spline[1],featimp_norm])

                # for features which are in space feature list but not in the other field's feature list
                for hf in her_features:
                    if hf not in field_feat:
                        field_her.append(['HER',hf,0])

        her_sorted = []

        for fh in field_her:
            fh[2] = str(fh[2])
            her_sorted.append(fh)


        f = open(path_to_store_normalised_feature_file,'w')

        for fh in field_her:
            f.write(','.join(fh)+'\n')

        f.close()

    def get_highest_mean(self):

    ##############
    # get the features with the highest mean score
    ##############

        lines1 = open(path_to_normalised_nb_file,'r').readlines()
        lines2 = open(path_to_normalised_sgd_file,'r').readlines()
        lines3 = open(path_to_normalised_extratree_file,'r').readlines()

        her = []

        for line in lines1:
            spline = line.replace('\n','').split(',')

            if spline[0] == 'HER':
                her.append([spline[1],spline[2]])

        for line in lines2:
            spline = line.replace('\n','').split(',')

            if spline[0] == 'HER':
                her.append([spline[1],spline[2]])

        for line in lines3:
            spline = line.replace('\n','').split(',')

            if spline[0] == 'HER':
                her.append([spline[1],spline[2]])


        feature_list = []

        for h in her:

            if h[0] not in feature_list:

                feature_list.append(h[0])


        feat_and_scores_all = []

        for fl in feature_list:

            feat_and_scores = []
            feat_and_scores.append(fl)

            for h in her:

                if h[0] == fl:
                    feat_and_scores.append(float(h[1]))

            feat_and_scores_all.append(feat_and_scores)


        # get mean feature score for each feature

        feat_mean = []

        for fs in feat_and_scores_all:

            mean = round((np.mean(fs[1:])),4)

            feat_mean.append([fs[0],mean])

        feat_mean.sort(key=lambda x: x[1], reverse=True)


        feat_mean_sorted = []
        top_feats = []

        for fm in feat_mean:
            fm[1] = str(fm[1])
            feat_mean_sorted.append([fm[0],fm[1]])

        # get the top ten features with highest mean score

        for fm in feat_mean[:15]:
            top_feats.append(fm[0])

        print (top_feats)

        f = open(path_to_store_feature_mean_score,'w')

        for fm in feat_mean_sorted:
            f.write(','.join(fm)+'\n')

        f.close()

        return top_feats



##############
# variables
##############

# change the FIELD for the following
path_to_nb_file = '../output/featimp_normalisation/nb/follcorr/nonprofit.csv'
path_to_sgd_file = '../output/featimp_normalisation/sgd/follcorr/nonprofit.csv'
path_to_extratree_file = '../output/featimp_normalisation/extratree/follcorr/nonprofit.csv'

# change the CLASSIFIER for the following
path_to_space_feature_score_file = '../output/featimp_normalisation/sgd/follcorr/space.csv'
path_to_politics_feature_score_file = '../output/featimp_normalisation/sgd/follcorr/politics.csv'
path_to_business_feature_score_file = '../output/featimp_normalisation/sgd/follcorr/business.csv'
path_to_nonprofit_feature_score_file = '../output/featimp_normalisation/sgd/follcorr/nonprofit.csv'

# change the CLASSIFIER and FIELD for the following
path_to_feature_score_file_cf = '../output/featimp_normalisation/sgd/follcorr/nonprofit.csv'
#path_to_store_normalised_feature_file = '../output/featimp_normalisation/extratree/others/politics_normalised.csv'
path_to_store_normalised_feature_file = '../output/featimp_normalisation/sgd/follcorr/per_field/nonprofit_normalised.csv'

#------------------

# change the FIELD for the following
path_to_normalised_nb_file = '../output/featimp_normalisation/nb/follcorr/per_field/nonprofit_normalised.csv'
path_to_normalised_sgd_file = '../output/featimp_normalisation/sgd/follcorr/per_field/nonprofit_normalised.csv'
path_to_normalised_extratree_file = '../output/featimp_normalisation/extratree/follcorr/per_field/nonprofit_normalised.csv'
path_to_store_feature_mean_score = '../output/spss/follcorr/feature_means/nonprofit_feature_mean.csv'




if __name__ == '__main__':

    ch = CalculateHighestMean()

    #############
    # run this first for all classifiers and fields
    #############

    ch.get_normalised_feat_score()


    #############
    # then run this for all fields
    #############

    #ch.get_highest_mean()

