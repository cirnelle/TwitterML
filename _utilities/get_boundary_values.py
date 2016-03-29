__author__ = 'yi-linghwong'

####################
# get the boundary value for demarcating between 1 and 0
# for LIWC categories
####################


import os
import sys
from matplotlib import pyplot as plt
import numpy as np


class GetBoundaryValues():

    def create_category_lists_summary_dimensions(self):

        lines = open(path_to_liwc_result_file,'r').readlines()

        analytic = []
        clout = []
        authentic = []
        tone = []

        for line in lines[:1]:
            spline = line.replace('\n','').split('\t')
            length = len(spline)

            for index,s in enumerate(spline):

                if s == 'Analytic':
                    analytic_index = index

                if s == 'Clout':
                    clout_index = index

                if s == 'Authentic':
                    authentic_index = index

                if s == 'Tone':
                    tone_index = index


        print ("Number of element per line is "+str(length))

        for line in lines[1:]:
            spline = line.replace('\n','').split('\t')

            if len(spline) == length:
                analytic.append(float(spline[analytic_index]))
                clout.append(float(spline[clout_index]))
                authentic.append(float(spline[authentic_index]))
                tone.append(float(spline[tone_index]))

            else:
                pass

        return analytic,clout,authentic,tone



    def get_boundary_values_summary_dimensions(self):

        lists = self.create_category_lists_summary_dimensions()
        percentage = 0.25

        print (len(lists))

        for index,l in enumerate(lists):

            # sort the list in descending order

            sorted_list = sorted(l, reverse=True)

            # get the index for top 25%

            percentile = int(percentage*len(sorted_list))

            list_percentile = sorted_list[:percentile]

            if index == 0:

                print ("Analytic top boundary value is "+str(list_percentile[-1]))

            if index == 1:

                print ("Clout top boundary value is "+str(list_percentile[-1]))

            if index == 2:

                print ("Authentic top boundary value is "+str(list_percentile[-1]))

            if index == 3:

                print ("Tone top boundary value is "+str(list_percentile[-1]))


        ##############
        # get bottom boundary values
        ##############

        print ("############################")


        for index,l in enumerate(lists):

            # sort the list in ascending order

            sorted_list = sorted(l)

            # get the index for top 25%

            percentile = int(percentage*len(sorted_list))

            list_percentile = sorted_list[:percentile]

            if index == 0:

                print ("Analytic bottom boundary value is "+str(list_percentile[-1]))

            if index == 1:

                print ("Clout bottom boundary value is "+str(list_percentile[-1]))

            if index == 2:

                print ("Authentic bottom boundary value is "+str(list_percentile[-1]))

            if index == 3:

                print ("Tone bottom boundary value is "+str(list_percentile[-1]))


    def create_category_lists_grammar(self):

    ################
    # features: sixltr (six letter words), word per sentence, punctuation (exclamation and question mark)
    ################

        lines = open(path_to_liwc_result_file,'r').readlines()

        sixltr = []
        wps = []
        exclam = []
        qmark = []

        for line in lines[:1]:
            spline = line.replace('\n','').split('\t')
            length = len(spline)

            for index,s in enumerate(spline):

                if s == 'Sixltr':
                    sixltr_index = index

                if s == 'WPS':
                    wps_index = index

                if s == 'Exclam':
                    exclam_index = index

                if s == 'QMark':
                    qmark_index = index


        print ("Number of element per line is "+str(length))

        for line in lines[2:]:
            spline = line.replace('\n','').split('\t')

            if len(spline) == length:
                sixltr.append(float(spline[sixltr_index]))
                wps.append(float(spline[wps_index]))
                exclam.append(float(spline[exclam_index]))
                qmark.append(float(spline[qmark_index]))

            else:
                pass

        return sixltr,wps,exclam,qmark


    def get_boundary_value_grammar(self):

    ################
    # features: sixltr (six letter words), word per sentence, punctuation (exclamation and question mark)
    ################

        lists = self.create_category_lists_grammar()
        percentage = 0.25

        print (len(lists))

        for index,l in enumerate(lists):

            # sort the list in descending order

            sorted_list = sorted(l, reverse=True)

            # get the index for top 25%

            percentile = int(percentage*len(sorted_list))

            list_percentile = sorted_list[:percentile]

            if index == 0:

                print ("Sixltr top boundary value is "+str(list_percentile[-1]))

            if index == 1:

                print ("WPS top boundary value is "+str(list_percentile[-1]))

            if index == 2:

                print ("Exclam top boundary value is "+str(list_percentile[-1]))

            if index == 3:

                print ("QMark top boundary value is "+str(list_percentile[-1]))


        ##############
        # get bottom boundary values
        ##############

        print ("############################")


        for index,l in enumerate(lists):

            # sort the list in ascending order

            sorted_list = sorted(l)

            # get the index for top 25%

            percentile = int(percentage*len(sorted_list))

            list_percentile = sorted_list[:percentile]

            if index == 0:

                print ("Sixltr bottom boundary value is "+str(list_percentile[-1]))

            if index == 1:

                print ("WPS bottom boundary value is "+str(list_percentile[-1]))

            if index == 2:

                print ("Exclam bottom boundary value is "+str(list_percentile[-1]))

            if index == 3:

                print ("QMark bottom boundary value is "+str(list_percentile[-1]))


    def plot_histogram(self):


        lists = self.create_category_lists_summary_dimensions()

        for index,l in enumerate(lists):

            plt.hist(l,bins=30)

            if index == 0:

                plt.xlabel("Analytic score")
                plt.ylabel("Number of posts")
                plt.show(block=True)

            if index == 1:

                plt.xlabel("Clout score")
                plt.ylabel("Number of posts")
                plt.show(block=True)

            if index == 2:

                plt.xlabel("Authentic score")
                plt.ylabel("Number of posts")
                plt.show(block=True)

            if index == 3:

                plt.xlabel("Tone score")
                plt.ylabel("Number of posts")
                plt.show(block=True)

        lists = self.create_category_lists_grammar()

        for index,l in enumerate(lists):

            plt.hist(l,bins=30)

            if index == 0:

                plt.xlabel("Sixltr score")
                plt.ylabel("Number of posts")
                plt.show(block=True)

            if index == 1:

                plt.xlabel("WPS score")
                plt.ylabel("Number of posts")
                plt.show(block=True)

            if index == 2:

                plt.xlabel("Exclam score")
                plt.ylabel("Number of posts")
                plt.show(block=True)

            if index == 3:

                plt.xlabel("QMark score")
                plt.ylabel("Number of posts")
                plt.show(block=True)


##############
# variables
##############

path_to_liwc_result_file = '../output/liwc/liwc_raw_politics.txt'


if __name__ == '__main__':

    
    gb = GetBoundaryValues()

    #gb.create_category_lists_summary_dimensions()
    gb.get_boundary_values_summary_dimensions()

    #gb.create_category_lists_grammar()
    gb.get_boundary_value_grammar()

    gb.plot_histogram()
