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

starttime=time.time()
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def lip_aspect_ratio(mouth):
    A= distance.euclidean(mouth[1], mouth[11])
    B= distance.euclidean(mouth[3], mouth[9])
    C= distance.euclidean(mouth[4], mouth[8])
    D= distance.euclidean(mouth[0], mouth[6])

    lar=(A+B+C)/(3.0*D)
    return lar
thresh = 0.26
frame_check = 10
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(sStart, sEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
cap=cv2.VideoCapture(0)
flag=0
l=[]
li=[]
try:
    
    while True:
        ret, frame=cap.read()
        frame = imutils.resize(frame, width=900)
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

            #plt.show()

            
            #face alignment
            leftEyeCenter = leftEye.mean(axis=0).astype("int")
            rightEyeCenter = rightEye.mean(axis=0).astype("int")
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            #print (angle)
            #lar
            lar=lip_aspect_ratio(mouth)
            dates=strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
            local_time=strftime("%H:%M:%S +0000", localtime())
        
            l.append([(datetime.now().strftime('%Y-%m-%d %H:%M:%S %f')),(datetime.now().strftime('%H:%M')),leftEAR,rightEAR,ear,lar,angle])            


            #print lar
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            #noseHull = cv2.convexHull(nose)
            #cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "Eye Aspect Ratio : {:.2f}".format(ear), (0, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
                        li.append([(datetime.now().strftime('%Y-%m-%d %H:%M:%S %f')),(datetime.now().strftime('%H:%M')),s,time.time(),(ree-s),flag])
                        flag = 0
                        print time.time()-s
                    else:
                        flag=0
        cv2.imshow("Frame", frame)
        if len(li)>10000:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(l,columns=["date_and_time","time","left_eye_EAR","right_eye_EAR","AVERAGE_EAR","LAR","ANGLE"])
    df1 = pd.DataFrame(li,columns=["date_and_time","time","blinking start time","blinking end time","blinking time","no of frames"])
    df.to_csv("data1.csv",sep="\t")
    df1.to_csv("blinking data.csv",sep="\t")
except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(l,columns=["date_and_time","time","left_eye_EAR","right_eye_EAR","AVERAGE_EAR","LAR","ANGLE"])
    df1 = pd.DataFrame(li,columns=["date_and_time","time","blinking start time","blinking end time","blinking time","no of frames"])
    df.to_csv("data1.csv",sep="\t")
    df1.to_csv("blinking data.csv",sep="\t")    
    