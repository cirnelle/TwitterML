__author__ = 'yi-linghwong'

import os
import sys
import numpy as np

class CreateSpssInput():

    def get_space_highest_mean(self):

    ##############
    # get the features with the highest mean score
    ##############

        lines1 = open(path_to_space_nb_file,'r').readlines()
        lines2 = open(path_to_space_sgd_file,'r').readlines()
        lines3 = open(path_to_space_extratree_file,'r').readlines()

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

        for fm in feat_mean[:10]:
            top_feats.append(fm[0])

        f = open(path_to_store_feature_mean_score,'w')

        for fm in feat_mean_sorted:
            f.write(','.join(fm)+'\n')

        f.close()

        return top_feats


    def get_only_her_or_ler(self,field):

        lines1 = open('../output/featimp_normalisation/nb/normalised_'+field+'.csv','r').readlines()
        lines2 = open('../output/featimp_normalisation/sgd/normalised_'+field+'.csv','r').readlines()
        lines3 = open('../output/featimp_normalisation/extratree/normalised_'+field+'.csv','r').readlines()

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

        return her

    def create_spss_input(self):

        space_top_feats = self.get_space_highest_mean()

        print (space_top_feats)

        space_her = self.get_only_her_or_ler('space')
        politics_her = self.get_only_her_or_ler('politics')
        business_her = self.get_only_her_or_ler('business')
        nonprofit_her = self.get_only_her_or_ler('nonprofit')

        for st in space_top_feats:

            feat = []

            for sh in space_her:
                if sh[0] == st:
                    feat.append([sh[0],'space',sh[1]])

            for ph in politics_her:
                if ph[0] == st:
                    feat.append([ph[0],'politics',ph[1]])

            for bh in business_her:
                if bh[0] == st:
                    feat.append([bh[0],'business',bh[1]])

            for nh in nonprofit_her:
                if nh[0] == st:
                    feat.append([nh[0],'nonprofit',nh[1]])

            print (len(feat))

            f = open('../output/spss/'+st+'.csv','w')

            for fe in feat:
                f.write(','.join(fe)+'\n')

            f.close()








##############
# variables
##############

path_to_space_nb_file = '../output/featimp_normalisation/nb/normalised_space.csv'
path_to_space_sgd_file = '../output/featimp_normalisation/sgd/normalised_space.csv'
path_to_space_extratree_file = '../output/featimp_normalisation/extratree/normalised_space.csv'

path_to_politics_nb_file = '../output/featimp_normalisation/nb/normalised_politics.csv'
path_to_politics_sgd_file = '../output/featimp_normalisation/sgd/normalised_politics.csv'
path_to_politics_extratree_file = '../output/featimp_normalisation/extratree/normalised_politics.csv'

path_to_business_nb_file = '../output/featimp_normalisation/nb/normalised_business.csv'
path_to_business_sgd_file = '../output/featimp_normalisation/sgd/normalised_business.csv'
path_to_business_extratree_file = '../output/featimp_normalisation/extratree/normalised_business.csv'

path_to_nonprofit_nb_file = '../output/featimp_normalisation/nb/normalised_nonprofit.csv'
path_to_nonprofit_sgd_file = '../output/featimp_normalisation/sgd/normalised_nonprofit.csv'
path_to_nonprofit_extratree_file = '../output/featimp_normalisation/extratree/normalised_nonprofits.csv'

path_to_store_feature_mean_score = '../output/spss/feature_mean.csv'





if __name__ == '__main__':

    cs = CreateSpssInput()

    #cs.get_space_highest_mean()
    #cs.get_only_her_or_ler()
    cs.create_spss_input()
