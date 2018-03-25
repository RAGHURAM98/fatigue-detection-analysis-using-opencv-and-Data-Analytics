# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:11:50 2018

@author: ram
"""

import matplotlib.pyplot as plt
import time
import numpy as np
fig=plt.figure()
plt.axis([0,1000,0,1])

i=0
x=list()
y=list()

while i <1000:
    temp_y=np.random.random()
    x.append(i)
    y.append(temp_y)
    plt.plot(i,temp_y)

    i+=1
plt.show()
#plt.close()