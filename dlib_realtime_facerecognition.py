# USAGE
# python dlib_realtime_facerecognition.py --landmark shape_predictor_5_face_landmarks.dat --model dlib_face_recognition_resnet_model_v1.dat --display 1
# python dlib_realtime_facerecognition.py --landmark shape_predictor_5_face_landmarks.dat --model dlib_face_recognition_resnet_model_v1.dat --display 0

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import numpy as np
import time
import cv2
import dlib
import os

# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--landmark", required=True,
	help="path to input landmark file")
ap.add_argument("-m", "--model", required=True,
	help="path to input model file")
ap.add_argument("-c", "--confidence", default=.5,
	help="confidence threshold")
ap.add_argument("-f", "--faces", required=True,
	help="path to known faces folder")
ap.add_argument("-d", "--display", type=int, default=1,
	help="switch to display image on screen")
args = vars(ap.parse_args())

# frame dimensions should be sqaure
PREPROCESS_DIMS = (640, 480)
DISPLAY_DIMS = (900, 900)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(args['landmark'])
facerec = dlib.face_recognition_model_v1(args['model'])

def load_known_faces(path2photos):
	known_dict = {}
	folders = os.listdir(path2photos)
	for folder in folders:
		files = os.listdir(path2photos + folder)
		face_desc = []
		for file in files:
			if os.path.splitext(file)[1] == '.jpg':
				img = cv2.imread(path2photos + folder + '/' + file)
				print("{}: {}*{}".format(file, img.shape[0], img.shape[1]))
				shape = sp(img, dlib.rectangle(0, 0, img.shape[0], img.shape[1]))
				face_descriptor = facerec.compute_face_descriptor(img, shape)
				face_desc.append(face_descriptor)
		known_dict[folder] = face_desc
	return known_dict

def find_known_face(face_dict, face):
	match_dict = {}
	for key, values in face_dict.items():
		confidence = 1.0
		for i, val in enumerate(values):
			diff = np.linalg.norm(np.array(face) - np.array(val))
			if diff < 1 - args['confidence'] and diff < confidence:
				confidence = diff
		match_dict[key] = confidence
	return sorted(match_dict.items(), key=lambda item: item[1])

def preprocess_image(input_image):
	# preprocess the image

	# return the image to the calling function
	return input_image


def predict(image, graph=None):
	# preprocess the image
	image = preprocess_image(image)
	predictions = []
	dets = detector(image, 1)
	print('detect {} faces'.format(len(dets)))
	for i, d in enumerate(dets):
		predictions.append((d.left(), d.top(), d.right(), d.bottom()))

	# return the list of predictions to the calling function
	return predictions


KNOWN_FACE_DICT = load_known_faces(args['faces'])
# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter
print("[INFO] starting the video stream and FPS counter...")
vs = VideoStream(src=0).start()
#vs = VideoStream(src="rtsp://admin:Baustem123@192.168.1.153:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1", framerate=1).start()
#time.sleep(1)
fps = FPS().start()

# loop over frames from the video file stream
while True:
	try:
		# grab the frame from the threaded video stream
		# make a copy of the frame and resize it for display/video purposes
		image_for_result = vs.read()
		#image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)

		predictions = predict(image_for_result)

		# loop over our predictions
		for i, (left, top, right, bottom) in enumerate(predictions):
			cv2.rectangle(image_for_result, (left, top), (right, bottom), (255, 0, 0), 2)
			shape = sp(image_for_result, dlib.rectangle(left, top, right, bottom))
			face_descriptor = facerec.compute_face_descriptor(image_for_result, shape)
			matched = find_known_face(KNOWN_FACE_DICT, face_descriptor)
			print matched
			for i in range(len(matched)):
				cv2.putText(image_for_result, "{}:{:.2f}".format(matched[i][0][0:2], 1 - matched[i][1]), (left, top - (40 - i*15)), cv2.FONT_HERSHEY_DUPLEX, 1,
							(255, 0, 0), 1)
				if i > 1:
					break

		# check if we should display the frame on the screen
		# with prediction data (you can achieve faster FPS if you
		# do not output to the screen)
		if args["display"] > 0:
			# display the frame to the screen
			fps.stop()
			cv2.putText(image_for_result, "FPS = {:.2f}".format(fps.fps()), (100,100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
			cv2.imshow("Output", image_for_result)
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
		# update the FPS counter
		fps.update()

	# if "ctrl+c" is pressed in the terminal, break from the loop
	except KeyboardInterrupt:
		break

	# if there's a problem reading a frame, break gracefully
	except AttributeError:
		break

# stop the FPS counter timer
fps.stop()

# destroy all windows if we are displaying them
if args["display"] > 0:
	cv2.destroyAllWindows()

# stop the video stream
vs.stop()
# clean up the graph and device

# display FPS information
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))