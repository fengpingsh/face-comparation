A very simple demo for face comparation and real time face recognition using dlib
=================================================================================

# Precondition
the dlib, imutils, opencv, numpy etc. are needed for loading videos, motion detection, face detection 
and facial comparation.

# Introduction
## About comparation.py
Comapre the two given faces(NOT pictures, the face was detected by default.), and return the confidence value, 
As described above, the face detection is not included, and about the result, you can find it in results folder. 
### Usage
python comparation.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat path2face1 path2face2 path2result
The shape_predictor_5_face_landmarks.dat and dlib_face_recognition_resnet_model_v1.dat was files pretrained by dlib. 
You can download them in http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 and 
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2.
And the path2face1, path2face2 and path2result can be specified when using this demo.

## About dlib_realtime_facerecognition.py
In this demo, I use a USB camera pluged in Ubuntu PC, and the imutils tool is used to get the real time video.
After got the real time video frame, I use the motion detection to find out the areas which is different with the former frame,
in this way, I can easily find the ROI which contain the faces. And then, the detection method in dlib is used to find the faces in the 
ROI, and the description of each detected faces are calculated. At last, the description is comapred to 
the known face descriptions which are previously loaded at the begining of the demo. The top 3 most likely canditates will be shown.

### Usage
--landmark: point to the file shape_predictor_5_face_landmarks.dat, also you can use the shape_predictor_68_face_landmarks.dat
--model: point to the dlib_face_recognition_resnet_model_v1.dat
--faces: point to the known faces
--confidence: confidence value to the known face, if it's too low, unknown faces may be treated as known.
--display: display the real time detection and recognition result or not.

Using the --faces param, you can specify the known faces. We suggest you place the front, left-turn, right-turn, looking-up and looking-down 
faces for each canditate.

During the running of this demo, you can press 'q' to quit and 's' to store the current frame. About the known faces generation, you can 
run the demo for a while, and let the candidates to the front of the camera, and the face pictures will automaticlly stored to the results
folder, you can pick up the faces you known, and place to the photos folder when their named folder.

You can change the PREPROCESS_DIMS for different resolution of camera to get a appropriate FPS. And also the threadhold for motion detection.

# TODO
Envelop it as a sensor, and provide the RESTFUL API to record faces and notify the results.
Group the denpendency in precondition to a single Requirement.txt file to simplify the installation.