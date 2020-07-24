#!/usr/bin/env python
# coding: utf-8

# In[52]:


import matplotlib.pyplot as plt
import numpy as np
import math

epsilon = 1.0e-3
lamb = 0.3
wave = np.array([[1., 0.], [0.25, 59.], [0.25, 127.], [0.25, 184.], [0.25, 221.], [0.25, 284.]])

def func(l, flag, x):
    sum_ = 0.
    if(flag == 'cos'):
        for i in l:
            sum_ += wave[i][0]*np.cos(-2*np.pi*x*np.cos(np.radians(wave[i][1])) / lamb)
    else:
        for i in l:
            sum_ += wave[i][0]*np.sin(-2*np.pi*x*np.cos(np.radians(wave[i][1])) / lamb)
    return sum_

x = np.arange(0, 5, 0.001)
y = np.array([10*math.log10(np.sqrt(func([0,1,2,3,4,5], 'cos', i)**2 + 
              func([0,1,2,3,4,5], 'sin', i)**2) + epsilon) for i in x])
plt.plot(x, y)
plt.savefig('D:/python/practice/r2_4.png')
plt.show()


# In[88]:


import matplotlib.pyplot as plt
from numpy.random import *
import numpy as np
import math
from scipy.special import *

fig = plt.figure(figsize=(18., 9.))
ax1, ax2 = fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)

epsilon = 1.e-3
wave = np.array([[1., 0.], [0.1, 59.], [0.1, 127.], [0.1, 184.], [0.1, 221.], [0,1, 284.]])
waves = [0,1,2,3,4,5] #合成する波のリスト
f = 1.  #(GHz)
lamb = 0.3  #(m)
T = 1./f
v = f*lamb

def func(l, flag, x):
    sum_ = 0.
    if(flag == 'cos'):
        for i in l:
            sum_ += wave[i][0]*np.cos(-2*np.pi*x*np.cos(np.radians(wave[i][1])) / lamb)
    else:
        for i in l:
            sum_ += wave[i][0]*np.sin(-2*np.pi*x*np.cos(np.radians(wave[i][1])) / lamb)
    return sum_

x = np.arange(0, 5, 0.01)
r_l = np.array([np.sqrt(func(waves, 'cos', i)**2 + 
                func(waves, 'sin', i)**2) for i in x])
ax1.hist(r_l, bins=80)
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
def rayleigh(r):
    return r/b*np.exp(-r**2/(2*b))

b = 0.
for i in waves:
    ai = -2*math.pi * v * math.cos(np.radians(wave[i][1])) / lamb
    print(ai)
    b += wave[i][0] / ai * (math.sin(np.radians(ai*T)) - math.sin(np.radians(0)))
    b -= wave[i][0] / ai * (math.cos(np.radians(ai*T)) - math.cos(np.radians(0)))
b = b/2./T

r1 = np.arange(0, np.max(r_l), 0.01)
pr1 = np.array([rayleigh(i) for i in r1])
#ax2.plot(r1, pr1)
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
def rician(r):
    return r/b * np.exp(-(r**2+A**2)/(2*b)) * jn(0, (r*A)/b)

A = 0.
b = 0.
for i in waves:
    ai = -2*math.pi * v * math.cos(np.radians(wave[i][1])) / lamb
    print(ai)
    b += wave[i][0] / ai * (math.sin(np.radians(ai*T)) - math.sin(np.radians(0)))
    b -= wave[i][0] / ai * (math.cos(np.radians(ai*T)) - math.cos(np.radians(0)))
b = b/2./T

r2 = np.arange(0, np.max(r_l), 0.01)
pr2 = np.array([rician(i) for i in r2])
ax2.plot(r2, pr2)
plt.savefig('D:/python/practice/r3_2.png')
plt.show()


# In[ ]:




