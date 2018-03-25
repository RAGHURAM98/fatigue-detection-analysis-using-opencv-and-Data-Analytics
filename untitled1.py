# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:46:38 2018

@author: ram
"""

import time
import pylab as pl
from IPython import display
for i in range(2):
    pl.plot(10)
    display.clear_output(wait=True)
    display.display(pl.gcf())