__author__ = 'yi-linghwong'

import os
import sys

class FeatureConstruction():

    def liwc_psychometric_features(self):

        lines = open(path_to_liwc_result_file,'r').readlines()

        print (len(lines))

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

                if s == 'posemo':
                    posemo_index = index

                if s == 'negemo':
                    negemo_index = index

                if s == 'anx':
                    anx_index = index

                if s == 'anger':
                    anger_index = index

                if s == 'sad':
                    sad_index = index

                if s == 'insight':
                    insight_index = index

                if s == 'cause':
                    cause_index = index

                if s == 'discrep':
                    discrep_index = index

                if s == 'tentat':
                    tentat_index = index

                if s == 'certain':
                    certain_index = index

                if s == 'differ':
                    differ_index = index

                if s == 'see':
                    see_index = index

                if s == 'hear':
                    hear_index = index

                if s == 'feel':
                    feel_index = index

                if s == 'affiliation':
                    affiliation_index = index

                if s == 'achieve':
                    achieve_index = index

                if s == 'power':
                    power_index = index

                if s == 'reward':
                    reward_index = index

                if s == 'risk':
                    risk_index = index

                if s == 'swear':
                    swear_index = index

                if s == 'netspeak':
                    netspeak_index = index

                if s == 'assent':
                    assent_index = index

                if s == 'nonflu':
                    nonflu_index = index

                if s == 'filler':
                    filler_index = index

        print ("Number of element per line is "+str(length))

        tweet_features = []

        for line in lines[1:]:
            spline = line.replace('\n','').split('\t')

            features = []

            if len(spline) == length:

                for n in range(28):

                    if n == 0:

                        if float(spline[analytic_index]) > analytic_top:
                            features.append('is_analytic_yes')

                        elif float(spline[analytic_index]) < analytic_bottom:
                            features.append('is_analytic_no')

                    if n == 1:

                        if float(spline[clout_index]) > clout_top:
                            features.append('is_clout_yes')

                        elif float(spline[clout_index]) < clout_bottom:
                            features.append('is_clout_no')

                    if n == 2:

                        if float(spline[authentic_index]) > authentic_top:
                            features.append('is_authentic_yes')

                        elif float(spline[authentic_index]) < authentic_bottom:
                            features.append('is_authentic_no')

                    if n == 3:

                        if float(spline[tone_index]) > tone_top:
                            features.append('is_tone_yes')

                        elif float(spline[tone_index]) < tone_bottom:
                            features.append('is_tone_no')

                    if n == 4:

                        if float(spline[posemo_index]) > 0.0:
                            features.append('posemo_yes')

                    if n == 5:

                        if float(spline[negemo_index]) > 0.0:
                            features.append('negemo_yes')

                    if n == 6:

                        if float(spline[anx_index]) > 0.0:
                            features.append('anx_yes')

                    if n == 7:

                        if float(spline[anger_index]) > 0.0:
                            features.append('anger_yes')

                    if n == 8:

                        if float(spline[sad_index]) > 0.0:
                            features.append('sad_yes')

                    if n == 9:

                        if float(spline[insight_index]) > 0.0:
                            features.append('insight_yes')

                    if n == 10:

                        if float(spline[cause_index]) > 0.0:
                            features.append('cause_yes')

                    if n == 11:

                        if float(spline[discrep_index]) > 0.0:
                            features.append('discrep_yes')

                    if n == 12:

                        if float(spline[tentat_index]) > 0.0:
                            features.append('tentat_yes')

                    if n == 13:

                        if float(spline[certain_index]) > 0.0:
                            features.append('certain_yes')

                    if n == 14:

                        if float(spline[differ_index]) > 0.0:
                            features.append('differ_yes')

                    if n == 15:

                        if float(spline[see_index]) > 0.0:
                            features.append('see_yes')

                    if n == 16:

                        if float(spline[hear_index]) > 0.0:
                            features.append('hear_yes')

                    if n == 17:

                        if float(spline[feel_index]) > 0.0:
                            features.append('feel_yes')

                    if n == 18:

                        if float(spline[affiliation_index]) > 0.0:
                            features.append('affiliation_yes')

                    if n == 19:

                        if float(spline[achieve_index]) > 0.0:
                            features.append('achieve_yes')

                    if n == 20:

                        if float(spline[power_index]) > 0.0:
                            features.append('power_yes')

                    if n == 21:

                        if float(spline[reward_index]) > 0.0:
                            features.append('reward_yes')

                    if n == 22:

                        if float(spline[risk_index]) > 0.0:
                            features.append('risk_yes')

                    if n == 23:

                        if float(spline[swear_index]) > 0.0:
                            features.append('swear_yes')

                    if n == 24:

                        if float(spline[netspeak_index]) > 0.0:
                            features.append('netspeak_yes')

                    if n == 25:

                        if float(spline[assent_index]) > 0.0:
                            features.append('assent_yes')

                    if n == 26:

                        if float(spline[nonflu_index]) > 0.0:
                            features.append('nonflu_yes')

                    if n == 27:

                        if float(spline[filler_index]) > 0.0:
                            features.append('filler_yes')


                if len(features) == 0:
                    print("No liwc feature for this tweet")
                    print (spline[0])
                    features.append('none')

                tweet_features.append(features)

            else:
                print ("Length of spline incorrect")
                print (len(spline),line)


        print (len(tweet_features))

        f = open(path_to_store_psychometric_feature_file,'w')

        for tf in tweet_features:
            f.write(' '.join(tf)+'\n')

        f.close()


    def liwc_grammar_features(self):

    ################
    # features: sixltr (six letter words), word per sentence, punctuation (exclamation and question mark)
    ################

        lines = open(path_to_liwc_result_file,'r').readlines()

        print (len(lines))

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

        tweet_features = []

        for line in lines[1:]:
            spline = line.replace('\n','').split('\t')

            features = []

            if len(spline) == length:

                for n in range(4):

                    if n == 0:

                        if float(spline[sixltr_index]) > sixltr_top:
                            features.append('many_sixltr')

                        elif float(spline[sixltr_index]) < sixltr_bottom:
                            features.append('few_sixltr')

                    if n == 1:

                        if float(spline[wps_index]) > wps_top:
                            features.append('high_wps')

                        elif float(spline[wps_index]) < wps_bottom:
                            features.append('low_wps')

                    if n == 2:

                        if float(spline[exclam_index]) > 0.0:
                            features.append('has_exclam')

                        else:
                            features.append('no_exclam')

                    if n == 3:

                        if float(spline[qmark_index]) > 0.0:
                            features.append('has_qmark')

                        else:
                            features.append('no_qmark')

                if len(features) == 0:
                    print("No grammar feature for this tweet")
                    print (spline[0])
                    features.append('none')

                tweet_features.append(features)

            else:
                print ("Length of spline incorrect")
                print (len(spline),line)


        print (len(tweet_features))

        f = open(path_to_store_grammar_feature_file,'w')

        for tf in tweet_features:
            f.write(' '.join(tf)+'\n')

        f.close()


    def url_hashtag_media_feature(self):

    ################
    # get url,hashtag and media (images or videos) features
    ################

        lines = open(path_to_labelled_raw_file,'r').readlines()

        tweets = []

        for line in lines:
            spline = line.replace('\n','').split(',')
            tweets.append(spline)

        print ("Length of tweets is "+str(len(tweets)))

        tweet_features = []
        url_list = []
        hashtag_list = []

        for t in tweets:
            features = []

            #append the type (i.e. photo, video, link, etc)

            if t[2] != 'has_no_media':

                features.append(t[2])

            ################
            #replace special character at end of sentence with white space
            #so that hashtags without a space in front of them can be detected too (e.g. This is it.#space)
            ################

            t[0] = t[0].replace(',',' ').replace('.',' ').replace('!',' ').replace('?',' ')

            ################
            # need two different loops and a 'break' after detecting the first symbol
            # so that there will be no repeated features when there are more than one hashtags/urls
            ################

            for word in t[0].split():
                if word.startswith("#"):
                    features.append("has_hashtag")
                    hashtag_list.append(word)
                    break

            for word in t[0].split():
                if word.startswith("http://") or word.startswith("https://"):
                    features.append("has_url")
                    url_list.append(word)
                    break


            if len(features) == 0:
                print("No sm feature for this tweet")
                print (t[0])
                features.append('none')


            features = ' '.join(features)

            # append the label to the list
            tweet_features.append([features,t[1]])

        print ("Length of tweet features list is "+str(len(tweet_features)))
        print ("Length of url list is "+str(len(url_list)))
        print ("Length of hashtag list is "+str(len(hashtag_list)))

        f = open(path_to_store_labelled_urlhashtagmedia_file,'w')

        for tf in tweet_features:
            f.write(','.join(tf)+'\n')

        f.close()


    def combine_features(self):

    ################
    # combine psychometrics, grammar and SM (url, hashtags, type) features
    ################

        lines = open(path_to_liwc_result_file,'r').readlines()

        print (len(lines))

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

                if s == 'posemo':
                    posemo_index = index

                if s == 'negemo':
                    negemo_index = index

                if s == 'anx':
                    anx_index = index

                if s == 'anger':
                    anger_index = index

                if s == 'sad':
                    sad_index = index

                if s == 'insight':
                    insight_index = index

                if s == 'cause':
                    cause_index = index

                if s == 'discrep':
                    discrep_index = index

                if s == 'tentat':
                    tentat_index = index

                if s == 'certain':
                    certain_index = index

                if s == 'differ':
                    differ_index = index

                if s == 'see':
                    see_index = index

                if s == 'hear':
                    hear_index = index

                if s == 'feel':
                    feel_index = index

                if s == 'affiliation':
                    affiliation_index = index

                if s == 'achieve':
                    achieve_index = index

                if s == 'power':
                    power_index = index

                if s == 'reward':
                    reward_index = index

                if s == 'risk':
                    risk_index = index

                if s == 'swear':
                    swear_index = index

                if s == 'netspeak':
                    netspeak_index = index

                if s == 'assent':
                    assent_index = index

                if s == 'nonflu':
                    nonflu_index = index

                if s == 'filler':
                    filler_index = index

                if s == 'Sixltr':
                    sixltr_index = index

                if s == 'WPS':
                    wps_index = index

                if s == 'Exclam':
                    exclam_index = index

                if s == 'QMark':
                    qmark_index = index

        print ("Number of element per line is "+str(length))

        tweet_features = []

        for line in lines[1:]:
            spline = line.replace('\n','').split('\t')

            features = []

            if len(spline) == length:

                for n in range(32):

                    if n == 0:

                        if float(spline[analytic_index]) > analytic_top:
                            features.append('is_analytic_yes')

                        elif float(spline[analytic_index]) < analytic_bottom:
                            features.append('is_analytic_no')

                    if n == 1:

                        if float(spline[clout_index]) > clout_top:
                            features.append('is_clout_yes')

                        elif float(spline[clout_index]) < clout_bottom:
                            features.append('is_clout_no')

                    if n == 2:

                        if float(spline[authentic_index]) > authentic_top:
                            features.append('is_authentic_yes')

                        elif float(spline[authentic_index]) < authentic_bottom:
                            features.append('is_authentic_no')

                    if n == 3:

                        if float(spline[tone_index]) > tone_top:
                            features.append('is_tone_yes')

                        elif float(spline[tone_index]) < tone_bottom:
                            features.append('is_tone_no')

                    if n == 4:

                        if float(spline[posemo_index]) > 0.0:
                            features.append('posemo_yes')

                    if n == 5:

                        if float(spline[negemo_index]) > 0.0:
                            features.append('negemo_yes')

                    if n == 6:

                        if float(spline[anx_index]) > 0.0:
                            features.append('anx_yes')

                    if n == 7:

                        if float(spline[anger_index]) > 0.0:
                            features.append('anger_yes')

                    if n == 8:

                        if float(spline[sad_index]) > 0.0:
                            features.append('sad_yes')

                    if n == 9:

                        if float(spline[insight_index]) > 0.0:
                            features.append('insight_yes')

                    if n == 10:

                        if float(spline[cause_index]) > 0.0:
                            features.append('cause_yes')

                    if n == 11:

                        if float(spline[discrep_index]) > 0.0:
                            features.append('discrep_yes')

                    if n == 12:

                        if float(spline[tentat_index]) > 0.0:
                            features.append('tentat_yes')

                    if n == 13:

                        if float(spline[certain_index]) > 0.0:
                            features.append('certain_yes')

                    if n == 14:

                        if float(spline[differ_index]) > 0.0:
                            features.append('differ_yes')

                    if n == 15:

                        if float(spline[see_index]) > 0.0:
                            features.append('see_yes')

                    if n == 16:

                        if float(spline[hear_index]) > 0.0:
                            features.append('hear_yes')

                    if n == 17:

                        if float(spline[feel_index]) > 0.0:
                            features.append('feel_yes')

                    if n == 18:

                        if float(spline[affiliation_index]) > 0.0:
                            features.append('affiliation_yes')

                    if n == 19:

                        if float(spline[achieve_index]) > 0.0:
                            features.append('achieve_yes')

                    if n == 20:

                        if float(spline[power_index]) > 0.0:
                            features.append('power_yes')

                    if n == 21:

                        if float(spline[reward_index]) > 0.0:
                            features.append('reward_yes')

                    if n == 22:

                        if float(spline[risk_index]) > 0.0:
                            features.append('risk_yes')

                    if n == 23:

                        if float(spline[swear_index]) > 0.0:
                            features.append('swear_yes')

                    if n == 24:

                        if float(spline[netspeak_index]) > 0.0:
                            features.append('netspeak_yes')

                    if n == 25:

                        if float(spline[assent_index]) > 0.0:
                            features.append('assent_yes')

                    if n == 26:

                        if float(spline[nonflu_index]) > 0.0:
                            features.append('nonflu_yes')

                    if n == 27:

                        if float(spline[filler_index]) > 0.0:
                            features.append('filler_yes')

                    if n == 28:

                        if float(spline[sixltr_index]) > sixltr_top:
                            features.append('many_sixltr')

                        elif float(spline[sixltr_index]) < sixltr_bottom:
                            features.append('few_sixltr')

                    if n == 29:

                        if float(spline[wps_index]) > wps_top:
                            features.append('high_wps')

                        elif float(spline[wps_index]) < wps_bottom:
                            features.append('low_wps')

                    if n == 30:

                        if float(spline[exclam_index]) > 0.0:
                            features.append('has_exclam')

                    if n == 31:

                        if float(spline[qmark_index]) > 0.0:
                            features.append('has_qmark')


                if len(features) == 0:
                    print("No liwc feature for this tweet")
                    print (spline[0])
                    features.append('none')

                features = ' '.join(features)

                tweet_features.append(features)

            else:
                print ("Length of spline incorrect")
                print (len(spline),line)


        ##############
        # get url hashtag type features
        ##############

        lines = open(path_to_labelled_raw_file,'r').readlines()

        tweets = []

        for line in lines:
            spline = line.replace('\n','').split(',')
            tweets.append(spline)

        print ("Length of tweets is "+str(len(tweets)))

        tweet_features_sm = []
        url_list = []
        hashtag_list = []

        for t in tweets:
            features_sm = []

            #append the type (i.e. photo, video, link, etc)

            if t[2] != 'has_no_media':

                features_sm.append(t[2])

            ################
            #replace special character at end of sentence with white space
            #so that hashtags without a space in front of them can be detected too (e.g. This is it.#space)
            ################

            t[0] = t[0].replace(',',' ').replace('.',' ').replace('!',' ').replace('?',' ')

            ################
            # need two different loops and a 'break' after detecting the first symbol
            # so that there will be no repeated features when there are more than one hashtags/urls
            ################

            for word in t[0].split():
                if word.startswith("#"):
                    features_sm.append("has_hashtag")
                    hashtag_list.append(word)
                    break

            for word in t[0].split():
                if word.startswith("http://") or word.startswith("https://"):
                    features_sm.append("has_url")
                    url_list.append(word)
                    break

            if len(features_sm) == 0:
                print("No SM feature for this post")
                print (t[0])
                features_sm.append('none')


            features_sm = ' '.join(features_sm)

            tweet_features_sm.append(features_sm)

        print ("Length of liwc feature list is "+str(len(tweet_features)))
        print ("Length of sm feature list is "+str(len(tweet_features_sm)))


        if len(tweet_features) == len(tweet_features_sm):

            zipped = zip(tweet_features,tweet_features_sm)

        else:
            print ("Length of both lists not equal, exiting...")
            sys.exit()

        combined_features = []

        for z in zipped:
            z = list(z)
            z = ' '.join(z)
            combined_features.append(z)

        print ("Length of combined list is "+str(len(combined_features)))


        f = open(path_to_store_combined_feature_file,'w')

        for cf in combined_features:
            f.write(cf+'\n')

        f.close()

        return combined_features


    def combine_features_all(self):

    ##################
    # combine all features: psychometrics, grammar, SM, words
    ##################

        language_features = self.combine_features()

        lines = open(path_to_labelled_preprocessed_file,'r').readlines()

        word_features = []

        for line in lines:
            spline = line.replace('\n','').split(',')
            word_features.append(spline[0])

        print ("############################")
        print ("Length of language feature list is "+str(len(language_features)))
        print ("Length of word feature list is "+str(len(word_features)))

        if len(language_features) == len(word_features):
            zipped = zip(language_features,word_features)

        else:
            print ("Length of lists not equal, exiting...")
            sys.exit()

        all_features = []

        for z in zipped:
            z = list(z)
            z = ' '.join(z)
            all_features.append(z)

        print ("Length of combined feature list is "+str(len(all_features)))


        f = open(path_to_store_combined_feature_all_file,'w')

        for af in all_features:
            f.write(af+'\n')

        f.close()


    def join_features_and_target(self):

        lines = open(path_to_labelled_raw_file,'r').readlines()

        label = []

        for line in lines:
            spline = line.replace('\n','').split(',')
            label.append(spline[1])

        print ("Length of label list is "+str(len(label)))

        ##############
        # select one of the features to compare
        ##############

        #lines2 = open(path_to_store_psychometric_feature_file,'r').readlines()
        #lines2 = open(path_to_store_grammar_feature_file,'r').readlines()
        #lines2 = open(path_to_store_combined_feature_file,'r').readlines()
        lines2 = open(path_to_store_combined_feature_all_file,'r').readlines()

        features = []

        for line in lines2:
            spline = line.replace('\n','')
            features.append(spline)

        print ("Length of feature list is "+str(len(features)))


        if len(label) == len(features):
            zipped_list = zip(features,label)

        else:
            print ("Lists have different lengths, exiting...")
            sys.exit()

        # zip both list together

        feature_and_label = []

        for zl in zipped_list:
            zl = list(zl)
            feature_and_label.append(zl)

        print ("Length of combined list is "+str(len(feature_and_label)))

        ###############
        # select one of the path to store result
        ###############

        #f = open(path_to_store_labelled_psychometric_file,'w')
        #f = open(path_to_store_labelled_grammar_file,'w')
        #f = open(path_to_store_labelled_combined_features_file,'w')
        f = open(path_to_store_labelled_combined_features_all_file,'w')

        # add header
        header = ['tweet','label']

        feature_and_label.insert(0,header)

        for fl in feature_and_label:
            f.write(','.join(fl)+'\n')

        f.close()




###############
# variables
###############

path_to_liwc_result_file = '../output/liwc/liwc_raw_space.txt'
path_to_labelled_raw_file = '../output/engrate/labelled_space_raw.csv'
path_to_labelled_preprocessed_file = '../output/engrate/labelled_space.csv'

path_to_store_psychometric_feature_file = '../output/features/space/psychometrics.txt'
path_to_store_grammar_feature_file = '../output/features/space/grammar.txt'
path_to_store_combined_feature_file = '../output/features/space/combined.txt'
path_to_store_combined_feature_all_file = '../output/features/space/combined_all.txt' #includes word features

path_to_store_labelled_psychometric_file = '../output/features/space/labelled_psychometrics.csv'
path_to_store_labelled_grammar_file = '../output/features/space/labelled_grammar.csv'
path_to_store_labelled_urlhashtagmedia_file = '../output/features/space/labelled_urlhashtagmedia.csv'
path_to_store_labelled_combined_features_file = '../output/features/space/labelled_combined.csv'
path_to_store_labelled_combined_features_all_file = '../output/features/space/labelled_combined_all.csv'


# boundary values

analytic_top = 98.0
analytic_bottom = 87.0
clout_top = 74.0
clout_bottom = 50.0
authentic_top = 48.0
authentic_bottom = 1.4
tone_top = 91.0
tone_bottom = 25.0
sixltr_top = 27.0
sixltr_bottom = 15.0
wps_top = 18.0
wps_bottom = 9.0


if __name__ == '__main__':

    fc = FeatureConstruction()

    #fc.liwc_psychometric_features()
    #fc.liwc_grammar_features()
    #fc.url_hashtag_media_feature()
    #fc.combine_features()
    fc.combine_features_all()

    fc.join_features_and_target()

