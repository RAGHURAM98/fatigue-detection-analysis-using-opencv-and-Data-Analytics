# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 18:30:46 2018

@author: ram
"""

import pandas as pd
import time
l=[[1,2],[4,5]]
df = pd.DataFrame(l,columns=["a","b"])
print (df)
from time import strftime,localtime

local_time=strftime("%H:%M:%S", localtime())
print local_time
from datetime import datetime
ls=[]

print (datetime.now().strftime('%Y-%m-%d %H:%M:%S %f'))
df = pd.DataFrame(l,columns=["date_and_time","time","left_eye_EAR","right_eye_EAR","AVERAGE_EAR","LAR","ANGLE"])