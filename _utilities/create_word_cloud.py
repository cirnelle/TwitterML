__author__ = 'yi-linghwong'

lines = open('../output/word_cloud/sgd_her_features','r').readlines()

word_list = []

for line in lines:
    spline = line.replace('\n','').split(',')
    word_list.append(spline[1])

word_list = list(reversed(word_list))


print (len(word_list))

word_cloud = []

for i,a in enumerate(word_list):

    for n in range(i+1):
        word_cloud.append(a)


word_cloud_final = ' '.join(word_cloud)

f = open('../output/word_cloud/word_cloud.txt','w')

for wc in word_cloud_final:
    f.write(wc)

f.close()






