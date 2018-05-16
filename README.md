A simple demo for face comparation and real time face recognition using dlib
=================================================================================

# Precondition
the dlib, imutils, opencv, numpy etc. are needed for loading videos, motion detection, face detection 
and facial comparation.

# Introduction
## About comparation.py
Comapre the two given faces(FACE ONLY pictures, the detection is not supported in this demo.), and return the confidence value, 
As described above, the face detection is not included, and about the comparation result, you can find it in the last param you specified. 
### Usage
python comparation.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat path2face1 path2face2 path2result
The shape_predictor_5_face_landmarks.dat and dlib_face_recognition_resnet_model_v1.dat are files pretrained by dlib. 
You can download them in http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 and 
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2.
And the path2face1 and path2face2 are the two faces to be compared, and path2result is the results compared which shows the confident value 
on top.

## About dlib_realtime_facerecognition.py
In this demo, I use an USB camera pluged to Ubuntu PC, and the imutils model is used to get the real time video.
After got the real time video frame, I use the motion detection in opencv to find out the areas which is different with the former frame,
in this way, I can easily find the ROI which contain the faces. And then, the detection method in dlib is used to find out the faces in the 
ROIs, then the description of each detected faces are calculated. At last, the descriptions are comapred to the known face descriptions 
which are previously loaded at the begining of the demo. The top 3 most likely canditates will be shown.

### Usage
--landmark: point to the file shape_predictor_5_face_landmarks.dat, also you can use the shape_predictor_68_face_landmarks.dat  
--model: point to the dlib_face_recognition_resnet_model_v1.dat  
--faces: point to the known faces  
--confidence: confidence value to the known face, if it's too low, unknown faces may be treated as known.  
--display: display the real time detection and recognition result or not.  

Using the --faces param, you can specify the known faces(aka the canditates). We suggest you place the front, left-turn, right-turn, looking-up 
and looking-down faces for each canditate.

During the running of this demo, you can press 'q' to quit and 's' to store the current frame.

#### Structure of the faces folder
.  
├── name1  
│   ├── down.jpg  
│   ├── front.jpg  
│   ├── left.jpg  
│   ├── right.jpg  
│   └── up.jpg  
└── name2  
    ├── down.jpg  
    ├── front.jpg  
    ├── left.jpg  
    ├── right.jpg  
    └── up.jpg  
You can place as many known faces as you wish.

#### Structure of the results folder
The detected faces will be stored to the results folder automaticlly, and the frames which have the ROIs, the faces and their names as well.
The face file is start with 'face_' and the frame file is start with 'frame'.
With the faces and frames, you can easily struct your photos folder with the faces stored in the result folder, and also find the results obviously.

#### Code change notes 
You can change the PREPROCESS_DIMS for different resolution of camera to get a appropriate FPS. And also the threadhold for motion detection.

# TODO
Envelop it as a sensor, and provide the RESTFUL API to record faces and notify the results.
Group the denpendency in precondition to a single Requirement.txt file to simplify the installation.
TBD