# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 22:52:05 2018

@author: ram
"""

from datetime import date
from datetime import datetime

import time
s=time.time()#print date.today()
print datetime.time(datetime.now())
print time.now()

time.sleep(1)
g=time.time()
print time.ctime()
print time.time()-s