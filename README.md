comparation.py: A very simple demo for face comparation using dlib
dlib_realtime_facerecognition.py: A real time face detection and recognition demo using dlib
=================================================================================

# Precondition
the dlib, imutils, opencv, numpy etc. are needed for loading videos, detection and comparation

# Introduction
## About comparation.py
Comapre the two given faces, and return the confidence value, and you can find the result in results. 
In this comparation, the detection is not included, so you have to get the faces prepared.

## About dlib_realtime_facerecognition.py
In this demo, I use a USB camera pluged in Ubuntu PC, and use imutils to get the real time video.
After got the real time video frame, the detection method in dlib is used to find the faces in the frame,
and then, the description of each detected face is calculated. At last, the description is comapred to 
the known face description load at the begining of the demo. The top 3 most likey canditates will be shown.

The canditates photos can be specified using the param --faces. You can point to a folder which contain the 
known faces in different sub folders in which only includes the faces. We suggest you place the front , 
left-turn, right-turn, looking-up and looking-down faces for each canditate.

#TODO