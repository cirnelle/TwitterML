__author__ = 'yi-linghwong'

import time
import matplotlib.pyplot as plt

a = "Jan 24 2016"

b = time.strptime(a,'%b %d %Y')

t_epoch = time.mktime(b)

print (t_epoch)

t2 = "Feb 6 2016"

t2b = time.strptime(t2,'%b %d %Y')

t2_epoch = time.mktime(t2b)

print (t2_epoch)

x_delta = t2_epoch - t_epoch

y_delta = 0.1849586*x_delta

print (y_delta)

y = 14553986 - y_delta

print (y)
