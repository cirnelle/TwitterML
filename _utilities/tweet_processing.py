__author__ = 'yi-linghwong'

###############
# methods for tweet preprocessing, includes:
# remove URL, RT, mention, special characters, stopwords, remove all of the above ('cleanup') plus single characters
# label tweets ('LRT', 'HRT', 'ART')
###############

import re
import os
import sys
import itertools
#from nltk.corpus import stopwords
from sklearn.feature_extraction import text

lines = open('../stopwords/stopwords.csv', 'r').readlines()

my_stopwords=[]
for line in lines:
    my_stopwords.append(line.replace("\n", ""))

stop_words = text.ENGLISH_STOP_WORDS.union(my_stopwords)


class TweetProcessing():

    def get_element_number_per_line(self):

        lines = open(path_to_raw_tweet_file,'r').readlines()

        for line in lines[:1]:
            spline = line.replace('\n','').split(',')

        length = len(spline)

        return length


    def remove_url_mention_hashtag(self):

        lines = open(path_to_raw_tweet_file,'r').readlines()

        tweets = []
        for line in lines:

            spline = line.replace('\n','').split(',')
            tweets.append(spline)

        tweet_list = []

        length = self.get_element_number_per_line()

        print ("Removing url, mentions and hashtags...")

        for t in tweets:

            if (len(t)) == length:

                t1 = t[-1]

                #remove URLs
                t2 = re.sub(r'(?:https?\://)\S+', '', t1)

                #remove mentions
                t3 = re.sub(r'(?:\@)\S+', '', t2)

                #remove hashtags (just the symbol, not the key word)

                t4 = re.sub(r"#","", t3).strip()

                t5 = t4.lower()

                t[-1] = ' '+t5+' '

                tweet_list.append(t)

            else:
                print ("error")
                print (t)

        print (len(tweet_list))

        return tweet_list


    def remove_punctuations(self):

        # Replace punctuation with white space, not nil! So that words won't join together when punctuation is removed

        tweets = self.remove_url_mention_hashtag()

        tweet_list = []

        print ("Removing punctuations ...")

        for t in tweets:

            #remove special characters
            t1 = re.sub("[^A-Za-z0-9]+",' ', t[-1])
            t[-1] = t1

            tweet_list.append(t)

        print (len(tweet_list))

        return tweet_list


    def expand_contractions(self):

        contractions_dict = {
            ' isn\'t ': ' is not ',
            ' isn’t ': ' is not ',
            ' isnt ': ' is not ',
            ' isn ': ' is not ',
            ' aren\'t ': ' are not ',
            ' aren’t ': ' are not ',
            ' arent ': ' are not ',
            ' aren ': ' are not ',
            ' wasn\'t ': ' was not ',
            ' wasn’t ': ' was not ',
            ' wasnt ': ' was not ',
            ' wasn ': ' was not ',
            ' weren\'t ': ' were not ',
            ' weren’t ': ' were not ',
            ' werent ': ' were not ',
            ' weren ': ' were not ',
            ' haven\'t ': ' have not ',
            ' haven’t ': ' have not ',
            ' havent ': ' have not ',
            ' haven ': ' have not ',
            ' hasn\'t ': ' has not ',
            ' hasn’t ': ' has not ',
            ' hasnt ': ' has not ',
            ' hasn ': ' has not ',
            ' hadn\'t ': ' had not ',
            ' hadn’t ': ' had not ',
            ' hadnt ': ' had not ',
            ' hadn ': ' had not ',
            ' won\'t ': ' will not ',
            ' won’t ': ' will not ',
            ' wouldn\'t ': ' would not ',
            ' wouldn’t ': ' would not ',
            ' wouldnt ': ' would not ',
            ' wouldn ': ' would not ',
            ' didn\'t ': ' did not ',
            ' didn’t ': ' did not ',
            ' didnt ': ' did not ',
            ' didn ': ' did not ',
            ' don\'t ': ' do not ',
            ' don’t ': ' do not ',
            ' dont ': ' do not ',
            ' don ': ' do not ',
            ' doesn\'t ': ' does not ',
            ' doesn’t ': ' does not ',
            ' doesnt ': ' does not ',
            ' doesn ': ' does not ',
            ' can\'t ': ' can not ',
            ' can’t ': ' can not ',
            ' cant ': ' can not ',
            ' couldn\'t ': ' could not ',
            ' couldn’t ': ' could not ',
            ' couldnt ': ' could not ',
            ' couldn ': ' could not ',
            ' shouldn\'t ': ' should not ',
            ' shouldn’t ': ' should not ',
            ' shouldnt ': ' should not ',
            ' shouldn ': ' should not ',
            ' mightn\'t ': ' might not ',
            ' mightn’t ': ' might not ',
            ' mightnt ': ' might not ',
            ' mightn ': ' might not ',
            ' mustn\'t ': ' must not ',
            ' mustn’t ': ' must not ',
            ' mustnt ': ' must not ',
            ' mustn ': ' must not ',
            ' shan\'t ': ' shall not ',
            ' shan’t ': ' shall not ',
            ' shant ': ' shall not ',
            ' shan ': ' shall not ',
        }

        tweets = self.remove_punctuations()
        tweet_list = []

        contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()), re.IGNORECASE)

        def replace(match):

            return contractions_dict[match.group(0).lower()]

        print ("Expanding contractions ...")

        for t in tweets:

            t1 = contractions_re.sub(replace, t[-1])
            t[-1] = t1
            tweet_list.append(t)

        print (len(tweet_list))

        return tweet_list


    def remove_stopwords(self):

        tweets = self.expand_contractions()

        tweet_list=[]

        print ("Removing stopwords ...")

        for t in tweets:
            no_stop=[] #important

            for w in t[-1].split():
                #remove single characters and stop words
                if (len(w.lower())>=2) and (w.lower() not in stop_words):
                    no_stop.append(w.lower())


                    #join the list of words together into a string
                    t[-1] = " ".join(no_stop)

            tweet_list.append(t)

        print (len(tweet_list))

        return tweet_list


    def remove_rt(self):

    #############
    # remove the term 'rt'
    #############

        tweets = self.remove_stopwords()

        tweet_list = []

        print ("Removing rt...")

        for t in tweets:

            # add blank space before and after tweet so that if sentence starts with rt it can be detected (e.g. 'rt @nasa blah blah')
            t1 = ' '+t[-1]+' '
            t2 = t1.replace(' rt ',' ')
            t[-1] = t2
            tweet_list.append(t)

        print (len(tweet_list))

        return tweet_list


    def remove_duplicate(self):

        tweets = self.remove_rt()

        tweet_list = []
        temp = []

        print ("Removing duplicates...")

        for t in tweets:
            if t[-1] not in temp:
                temp.append(t[-1])
                tweet_list.append(t)

        print (len(tweet_list))

        return tweet_list


    def write_to_file(self):

        tweets = self.remove_rt()
        length = self.get_element_number_per_line()

        print ("Number of element per line is "+str(length))

        f = open(path_to_store_processed_tweet_file,'w')

        print ("Writing to file ...")

        for t in tweets:
            if (len(t)) == length:

                f.write(','.join(t)+'\n')

            else:
                print ("error")
                print (t)

        f.close()

        return


###############
# variables
###############

path_to_raw_tweet_file = '../tweets/raw_space_20160304.csv'
path_to_store_processed_tweet_file = '../tweets/preprocessed_space_20160304.csv'

if __name__ == "__main__":

    tp = TweetProcessing()

    tp.write_to_file()





