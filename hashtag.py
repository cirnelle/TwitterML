__author__ = 'yi-linghwong'

from extractor import Extractor

hashtaglist = ["sydsciencefest"]

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


