__author__ = 'yi-linghwong'


###############
# compute a updated follower count (extrapolation based on a linear equation)
# for a list of tweet
###############


import os
import sys
import time


class UpdateFollowerCount():


    def create_tweet_list(self):

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

                foll_count = self.compute_follower_count(t[0],t[1],t_max,t[2])

                if foll_count <= 0:

                    foll_count = follower_count[- 1]
                    t[2] = str(foll_count)
                    updated_tweets.append(t)

                else:
                    follower_count.append(foll_count)
                    t[2] = str(foll_count)
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


if __name__ == "__main__":

    path_to_tweet_file = '../extracted_data/ALL_nofollcorr.csv'
    path_to_slope_file = '../followers/user_slope_space.txt'
    path_to_store_updated_tweet_file = '../extracted_data/space_follcorr.csv'

    ################
    # create slope dict
    ################

    lines = open(path_to_slope_file,'r').readlines()

    slope_dict = {}

    for line in lines:
        spline = line.replace('\n','').split(',')
        slope_dict[spline[0]] = spline[1]

    print ("Length of slope_dict is "+str(len(slope_dict)))

    uf = UpdateFollowerCount()
    uf.create_tweet_list()