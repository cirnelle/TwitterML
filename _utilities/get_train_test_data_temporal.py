__author__ = 'yi-linghwong'

import os
import sys
import time


def get_train_test_temporal_raw():

    lines = open(path_to_labelled_file_raw_with_dates, 'r').readlines()
    print ("Total number of tweets: "+str(len(lines)))

    tweet_dates_her = []
    tweet_dates_ler = []

    for line in lines:

        spline = line.rstrip('\n').split(',')

        tweet_date = spline[0].split(' ')

        date = tweet_date[1] + ' ' + tweet_date[2] + ' ' + tweet_date[5] + ' ' + tweet_date[3]

        t1 = time.strptime(date, '%b %d %Y %H:%M:%S')

        t_epoch = time.mktime(t1)

        spline[0] = t_epoch

        if spline[2] == 'HER':

            tweet_dates_her.append(spline)

        if spline[2] == 'LER':

            tweet_dates_ler.append(spline)

    tweet_dates_her.sort(key=lambda x:x[0])
    tweet_dates_ler.sort(key=lambda x:x[0])

    print (len(tweet_dates_her))
    print (len(tweet_dates_ler))

    training_max_index_her = int(0.8 * len(tweet_dates_her))
    training_data_her = tweet_dates_her[:training_max_index_her]
    test_start_index_her = training_max_index_her
    test_data_her = tweet_dates_her[test_start_index_her:]

    training_max_index_ler = int(0.8 * len(tweet_dates_ler))
    training_data_ler = tweet_dates_ler[:training_max_index_ler]
    test_start_index_ler = training_max_index_ler
    test_data_ler = tweet_dates_ler[test_start_index_ler:]

    print ("Length of training data HER: "+str(len(training_data_her)))
    print ("Length of test data HER: "+str(len(test_data_her)))
    print ("Length of training data LER: "+str(len(training_data_ler)))
    print ("Length of test data LER: "+str(len(test_data_ler)))
    print ("Total: "+str(len(training_data_her)+len(test_data_her)+len(training_data_ler)+len(test_data_ler)))

    training_data = training_data_her + training_data_ler
    test_data = test_data_her + test_data_ler

    f = open(path_to_store_raw_training_data_temporal,'w')

    for tr_d in training_data:
        f.write(','.join(tr_d[1:])+'\n')

    f.close()

    f = open(path_to_store_raw_test_data_temporal,'w')

    for te_d in test_data:
        f.write(','.join(te_d[1:])+'\n')

    f.close()


def get_train_test_temporal_preprocessed():

    lines = open(path_to_labelled_file_preprocessed_with_dates, 'r').readlines()

    print()
    print("Total number of tweets: " + str(len(lines)))

    tweet_dates_her = []
    tweet_dates_ler = []

    for line in lines:

        spline = line.rstrip('\n').split(',')

        tweet_date = spline[0].split(' ')

        date = tweet_date[1] + ' ' + tweet_date[2] + ' ' + tweet_date[5] + ' ' + tweet_date[3]

        t1 = time.strptime(date, '%b %d %Y %H:%M:%S')

        t_epoch = time.mktime(t1)

        spline[0] = t_epoch

        if spline[2] == 'HER':
            tweet_dates_her.append(spline)

        if spline[2] == 'LER':
            tweet_dates_ler.append(spline)

    tweet_dates_her.sort(key=lambda x: x[0])
    tweet_dates_ler.sort(key=lambda x: x[0])

    print(len(tweet_dates_her))
    print(len(tweet_dates_ler))

    training_max_index_her = int(0.8 * len(tweet_dates_her))
    training_data_her = tweet_dates_her[:training_max_index_her]
    test_start_index_her = training_max_index_her
    test_data_her = tweet_dates_her[test_start_index_her:]

    training_max_index_ler = int(0.8 * len(tweet_dates_ler))
    training_data_ler = tweet_dates_ler[:training_max_index_ler]
    test_start_index_ler = training_max_index_ler
    test_data_ler = tweet_dates_ler[test_start_index_ler:]

    print("Length of training data HER: " + str(len(training_data_her)))
    print("Length of test data HER: " + str(len(test_data_her)))
    print("Length of training data LER: " + str(len(training_data_ler)))
    print("Length of test data LER: " + str(len(test_data_ler)))
    print("Total: " + str(len(training_data_her) + len(test_data_her) + len(training_data_ler) + len(test_data_ler)))

    training_data = training_data_her + training_data_ler
    test_data = test_data_her + test_data_ler

    f = open(path_to_store_preprocessed_training_data_temporal, 'w')

    for tr_d in training_data:
        f.write(','.join(tr_d[1:]) + '\n')

    f.close()

    f = open(path_to_store_preprocessed_test_data_temporal, 'w')

    for te_d in test_data:
        f.write(','.join(te_d[1:]) + '\n')

    f.close()


#############
# variables
#############

path_to_labelled_file_raw_with_dates = '../output/engrate/nonprofit/temporal/labelled_nonprofit_raw_dates.csv'
path_to_labelled_file_preprocessed_with_dates = '../output/engrate/nonprofit/temporal/labelled_nonprofit_dates.csv'

path_to_store_raw_training_data_temporal = '../output/engrate/nonprofit/temporal/training_nonprofit_raw.csv'
path_to_store_raw_test_data_temporal = '../output/engrate/nonprofit/temporal/test_nonprofit_raw.csv'
path_to_store_preprocessed_training_data_temporal = '../output/engrate/nonprofit/temporal/training_nonprofit.csv'
path_to_store_preprocessed_test_data_temporal = '../output/engrate/nonprofit/temporal/test_nonprofit.csv'


get_train_test_temporal_raw()
get_train_test_temporal_preprocessed()