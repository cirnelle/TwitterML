__author__ = 'yi-linghwong'

import sys
import os
from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import interp1d


lines = open('followers/follcount_NASA_no_inter.csv','r').readlines()

follcount_list = []


for line in lines:
    spline = line.replace('\n','').split(',')

    follcount_list.append(float(spline[1]))

print (len(follcount_list))


x = range (0,39)

print (len(x))



coefficients = np.polyfit(x, follcount_list, 1)
polynomial = np.poly1d(coefficients)
ys = polynomial(x)
print (coefficients)
print (polynomial)

plt.plot(x, follcount_list, 'o')
plt.plot(x, ys)

print("Linregress is "+str(linregress(x, ys)))

plt.show()














