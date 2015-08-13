__author__ = 'yi-linghwong'


lines = open('output/sydscifest/sydsciencefest_TMP.csv', 'r').readlines()
f = open('output/sydscifest/sdysciencefest_CLEAN.csv', 'w')

tweets = []

for line in lines:
    spline=line.replace("\n", "").split(",")

    if spline[3] not in tweets:
        f.write(line)
        tweets.append(spline[3])









