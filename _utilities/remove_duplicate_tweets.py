__author__ = 'yi-linghwong'

################
# method to remove duplicate tweets
###############


lines = open('output/sydscifest/ssf15_TMP.csv', 'r').readlines()
f = open('output/sydscifest/ssf15_CLEAN.csv', 'w')

tweets = []

for line in lines:
    spline=line.replace("\n", "").split(",")

    if spline[3] not in tweets:
        f.write(line)
        tweets.append(spline[3])









