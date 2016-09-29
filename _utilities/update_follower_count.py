__author__ = 'yi-linghwong'


###############
# compute a updated follower count (extrapolation based on a linear equation)
# for a list of tweet
###############


import os
import sys
import time


class UpdateFollowerCount():


    def update_tweet_list(self):

        ###############
        # create tweet list with updated follower count
        ###############

        lines = open(path_to_tweet_file,'r').readlines()

        tweets = []
        for line in lines:
            spline = line.replace('\n','').split(',')
            tweets.append(spline)

        print ("Length of tweet list is "+str(len(tweets)))

        unique_user = []
        updated_tweets = []
        follower_count = []

        for t in tweets:

            key = t[0]

            if key in slope_dict:

                if t[0] not in unique_user:
                    unique_user.append(t[0])
                    t1 = t[1].split(' ')
                    t2 = t1[1]+' '+t1[2]+' '+t1[5]
                    t3 = time.strptime(t2,'%b %d %Y')
                    t_max = time.mktime(t3)


                foll_count = self.compute_follower_count(t[0],t[1],t_max,t[3])

                #if foll_count <= 0:
                if foll_count < (0.09 * float(t[3])): # cap the minimum foll count to 2% of max follcount (to avoid engrate being disproportionately big for older tweets)

                    foll_count = follower_count[-1]
                    t[3] = str(foll_count)
                    updated_tweets.append(t)

                else:
                    follower_count.append(foll_count)
                    t[3] = str(foll_count)
                    updated_tweets.append(t)

            else:
                #updated_tweets.append(t)
                pass

        print ("Length of updated tweet list is "+str(len(updated_tweets)))

        f = open(path_to_store_updated_tweet_file,'w')

        for ut in updated_tweets:
            f.write(','.join(ut)+'\n')

        f.close()

        return updated_tweets



    def compute_follower_count(self,user,date,t_max,foll_count_max):

        ###############
        # function to compute follower count, taking the tweet date and latest follower count as arguments
        ###############


        # split the date by blank space
        # input = 'Thu Aug 06 22:18:01 +0000 2015'
        # result = ['Thu', 'Aug', '06', '22:18:01', '+0000', '2015']


        d1 = date.split(' ')

        d2 = d1[1]+' '+d1[2]+' '+d1[5]

        t1 = time.strptime(d2,'%b %d %Y')
        t_epoch = time.mktime(t1)

        t_delta = t_max - t_epoch

        slope = float(slope_dict[user])

        y_delta = slope*t_delta

        foll_count = float(foll_count_max) - y_delta

        foll_count = int(foll_count)

        #print ("Updated foll count is "+str(int(foll_count)))

        return foll_count


    def update_follcount_with_real_numbers(self):

        lines1 = open(path_to_tweet_file,'r').readlines()

        users = []

        for line in lines1[1:]:
            spline = line.rstrip('\n').split(',')

            if spline[0] not in users:
                users.append(spline[0])

        updated_tweets = []

        for u in users:

            print (u)

            ##################
            # get the dates for follcount file
            ##################

            lines2 = open(path_to_follcount_files+u+'.csv')

            date_dict = {}

            for line in lines2:
                spline = line.rstrip('\n').split(',')

                if spline == ['']:
                    print ("error")
                    break

                d1 = spline[0].replace('\n', '').split(' ')

                if len(d1) == 6:
                    d1.remove(d1[2])

                date_s = d1[1] + ' ' + d1[2] + ' ' + d1[4]

                t1 = time.strptime(date_s, '%b %d %Y')
                t_epoch = time.mktime(t1)
                date_dict[t_epoch] = spline[1]


            #################
            # get the dates for raw tweet file
            #################

            user_tweets = []

            for line in lines1[1:]:
                spline = line.rstrip('\n').split(',')

                if spline[0] == u:

                    d1 = spline[1].replace('\n', '').split(' ')

                    if len(d1) == 7:
                        print ("error")
                        d1.remove(d1[2])

                    date_s = d1[1] + ' ' + d1[2] + ' ' + d1[5]

                    t1 = time.strptime(date_s, '%b %d %Y')

                    t_epoch = time.mktime(t1)

                    if t_epoch in date_dict:

                        if date_dict[t_epoch] != 'nan':
                            spline[3] = date_dict[t_epoch]
                            user_tweets.append(spline)

                            updated_tweets.append(spline)

            print (len(user_tweets))


        print (" ")

        print (len(updated_tweets))

        f = open(path_to_store_realfoll_tweet_file,'w')

        for ut in updated_tweets:
            f.write(','.join(ut)+'\n')

        f.close()



###############
# variables
###############

path_to_tweet_file = '../tweets/others/raw_nonprofit.csv'
path_to_slope_file = '../followers/slope/user_slope_space.txt'
path_to_store_updated_tweet_file = '../tweets/follcorr/others/raw_space_follcorr.csv'

path_to_follcount_files = '../followers/follcount/follcount_'
path_to_store_realfoll_tweet_file = '../tweets/realfoll/raw_nonprofit_realfoll.csv'



if __name__ == "__main__":


    # ################
    # # create slope dict and update follcount
    # ################
    #
    # lines = open(path_to_slope_file,'r').readlines()
    #
    # slope_dict = {}
    #
    # for line in lines:
    #     spline = line.replace('\n','').split(',')
    #     slope_dict[spline[0]] = spline[1]
    #
    # print ("Length of slope_dict is "+str(len(slope_dict)))
    #
    # uf = UpdateFollowerCount()
    # uf.update_tweet_list()



    #################
    # update follcount with real data mined daily
    #################

    uf = UpdateFollowerCount()
    uf.update_follcount_with_real_numbers()
