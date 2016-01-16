__author__ = 'yi-linghwong'

a = [1,2,3]

print (range(len(a)))

for index in range(len(a)):

    print (index,a[index])


for index,value in enumerate(a):
    print (index,value)