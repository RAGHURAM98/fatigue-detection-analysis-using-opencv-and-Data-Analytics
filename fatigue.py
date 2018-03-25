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
import matplotlib.pyplot as plt


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
 
thresh = 0.25
frame_check = 10
try:
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
except:
    print "predictor file is not available at the location. pls check"
    
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(sStart, sEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cap=cv2.VideoCapture(0)
flag=0
fig = plt.figure()
l=[]
while True:
    ret, frame=cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)#converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth=shape[sStart:sEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        lar=lip_aspect_ratio(mouth)
        print lar
        ear = (leftEAR + rightEAR) / 2.0
        l.append(ear)
        plt.plot(l)
        plt.show()
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:
            flag += 1
            print (flag)
            if flag==1:
                s=time.time()
            
            if flag >= frame_check:
                cv2.putText(frame, "****************DROWSINESS ALERT!****************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if ear>thresh:
                if flag>=1:
                    #print "dhd"
                    print flag
                    flag = 0
                    print time.time()-s
                else:
                    #print 0
                    flag=0

    cv2.imshow("Frame", frame)
cv2.destroyAllWindows()
cap.stop()