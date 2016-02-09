__author__ = 'yi-linghwong'

##############
# get tweets by hashtags
##############

from extractor import Extractor


hashtaglist = ["mars water", "MarsAnnouncement"]

class UserTweets():


    def get_user_tweets(self,hashtaglist):


        ext= Extractor()
        auth = ext.loadtokens()
        api = ext.connectToAPI(auth)

        for ht in hashtaglist:

            full_tweets = ext.gettweets_hashtag(ht,api)
            ext.printcsv(full_tweets,ht)


ut = UserTweets()
ut.get_user_tweets(hashtaglist)

