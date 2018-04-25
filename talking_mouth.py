# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 04:07:43 2018

@author: ram
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 22:32:10 2018

@author: ram
"""
import pandas as pd
import re
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time
from datetime import date
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from time import strftime,localtime
from collections import Counter
import pylab

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return round(ear,2)

def lip_aspect_ratio(mouth):
    a= distance.euclidean(mouth[1], mouth[11])
    b=  distance.euclidean(mouth[2], mouth[10])
    c= distance.euclidean(mouth[3], mouth[9])
    d=distance.euclidean(mouth[4], mouth[8])
    e= distance.euclidean(mouth[5], mouth[7])
    f= distance.euclidean(mouth[0], mouth[6])
    g=distance.euclidean(mouth[12], mouth[16])
    h=distance.euclidean(mouth[13], mouth[19])
    i=distance.euclidean(mouth[14], mouth[18])
    j=distance.euclidean(mouth[15], mouth[17])
    #print g
    ular=(a+b+c+d+e)/(5.0*f)
    llar=(h)/(g)
    #print llar
    #print h,i
    return round(ular,3)

#predictor algorithm
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(sStart, sEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

#main code and start video capturing


lip=[]
#lists
w=[]
q=[]
lo=[]
l=[]
li=[]
lis=[]
lists=[]
listss=[]
listss.append(0.3)
lips=[]
#try and exception thing
cap=cv2.VideoCapture(0)
flag=0
flags=0
flag_check=0
que=0
thresh = 0.22
frame_check = 48
lip_check=0
lip_flag=0
aspect_ratio=400
try:
    while True:
        ret, frame=cap.read()
        frame = imutils.resize(frame, width=400)    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)#converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            nose = shape[nStart:nEnd]
            mouth=shape[sStart:sEnd]
            #eyes ear
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            #face alignment
            leftEyeCenter = leftEye.mean(axis=0).astype("int")
            rightEyeCenter = rightEye.mean(axis=0).astype("int")
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            #print (angle)
            #lar
            lar=lip_aspect_ratio(mouth)
            dates=(datetime.now().strftime('%Y-%m-%d %H:%M:%S %f'))
            times=(datetime.now().strftime('%H:%M'))
            seconds=(datetime.now().strftime('%H:%M:%S'))

            l.append([dates,times,leftEAR,rightEAR,ear,lar,angle])  
            lis.append([leftEAR,rightEAR])
            #lists.append([dates,times,a,b,c,d,e,f,lar])
            #listss.append([a,b,c])
            listss.append(lar)
            if lar-listss[len(listss)-2]>0.04 or listss[len(listss)-2]-lar>0.04 :
                lists.append([seconds,1,lip_check])
                lip_check+=1
            else:
                lip_check-=1
                lists.append([seconds,0,lip_check])
                
            if seconds!=lists[len(lists)-2][0]:
                lips.append([seconds,lip_check])
                #print lip_check
                if lip_check>0:
                    #lip_flag+=1
                    cv2.putText(frame, "****************LIP MOVEMENT ALERT!****************", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                lip_check=0
                #print seconds
            #if lip_flag>2:
               
                
        

            #print lar
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            #noseHull = cv2.convexHull(nose)
            #cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
            
            cv2.putText(frame, "left Ear : {:.2f} right ear{:.2f} average ear{:.2f} lar{:.3f} angle{:.3f}".format(leftEAR,rightEAR,ear,lar,angle), (0, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            if (leftEAR/1.0 - thresh<=0 ) & (rightEAR/1.0 - thresh<=0):
                flag += 1
                #print (flag)
                if flag==1:
                    s=time.time()
                if flag >= frame_check:
                    cv2.putText(frame, "****************DROWSINESS ALERT!****************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if (leftEAR/1.0 - thresh>0 ) & (rightEAR/1.0 - thresh > 0):
                    if flag>=1:
                        ree=time.time()
                        li.append([dates,times,s,time.time(),(ree-s),flag])
                        flag = 0
                        lo.append(times)
                        counts = Counter(lo).items()
                        #print(counts)
                        for i in range(0,len(counts)):
                            if times not in counts[i][0]:
                                if counts[i][0] not in w:
                                    w.append(counts[i][0])
                                    q.append([counts[i][0],counts[i][1]])
                                    if counts[i][1]>10:
                                        print "more blinks have been found",format(counts[i][0])
                                        flags+=1

                                    #print times
                                if flags>1 or que>2:
                                    print " fatigue"

                        if (ree-s)>1.0:
                            que+=1
                            #print que

                    else:
                        flag=0

        cv2.imshow("Frame", frame)
        cv2.imshow("gray",gray)
        key = cv2.waitKey(1) & 0xFF
        if len(l)%100==0:
            print len(l)
        if len(l)>40000:
            break
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(l,columns=["date_and_time","time","left_eye_EAR","right_eye_EAR","AVERAGE_EAR","LAR","ANGLE"])
    df1 = pd.DataFrame(li,columns=["date_and_time","time","blinking start time","blinking end time","blinking time","no of frames"])
    df.to_csv("data2.csv",sep="\t")
    df1.to_csv("blinking data2.csv",sep="\t")
    df3 = pd.DataFrame(lips,columns=["times","lip_movemnt_check"])
    df2 = pd.DataFrame(q,columns=["times","no of blinks per minute"])
    df2.to_csv("bink data.csv",sep="\t")
    df3.to_csv("lip tracking",sep="\t")


except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(l,columns=["date_and_time","time","left_eye_EAR","right_eye_EAR","AVERAGE_EAR","LAR","ANGLE"])
    df1 = pd.DataFrame(li,columns=["date_and_time","time","blinking start time","blinking end time","blinking time","no of frames"])
    df2 = pd.DataFrame(q,columns=["times","no of blinks per minute"])
    df3 = pd.DataFrame(lips,columns=["times","lip_movemnt_check"])
    df.to_csv("data2.csv",sep="\t")
    df1.to_csv("blinking data2.csv",sep="\t")  
    df2.to_csv("bink data.csv",sep="\t")
    df3.to_csv("lip tracking",sep="\t")

    