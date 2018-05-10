# USAGE
# python dlib_realtime_facerecognition.py --landmark shape_predictor_5_face_landmarks.dat --model dlib_face_recognition_resnet_model_v1.dat --faces path2faces folder --display 1
# python dlib_realtime_facerecognition.py --landmark shape_predictor_5_face_landmarks.dat --model dlib_face_recognition_resnet_model_v1.dat --faces path2faces folder --display 0

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
PREPROCESS_DIMS = 2
DISPLAY_DIMS = (1280, 720)

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
	#image = preprocess_image(image)
	predictions = []
	dets = detector(image, 1)
	#print('detect {} faces'.format(len(dets)))
	for i, d in enumerate(dets):
		predictions.append((d.left(), d.top(), d.right(), d.bottom()))

	# return the list of predictions to the calling function
	return predictions
avg = None
def motion(gray):
	_motions = []
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

	# threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, 4, 255,
						   cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
									cv2.CHAIN_APPROX_SIMPLE)
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 8000:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		_motions.append((x,y, x + w, y + h))
		#cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
	return _motions
KNOWN_FACE_DICT = load_known_faces(args['faces'])
# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter
print("[INFO] starting the video stream and FPS counter...")
#vs = VideoStream(src=0).start()
vs = VideoStream(src="rtsp://admin:Baustem123@192.168.3.22:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1").start()
#time.sleep(1)
fps = FPS().start()

# loop over frames from the video file stream
while True:
	try:
		# grab the frame from the threaded video stream
		# make a copy of the frame and resize it for display/video purposes
		frame = vs.read()

		image_for_result = cv2.resize(frame, (0,0), fx=1.0/PREPROCESS_DIMS, fy=1.0/PREPROCESS_DIMS)
		gray = cv2.cvtColor(image_for_result, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		# if the average frame is None, initialize it
		if avg is None:
			print "[INFO] starting background model..."
			avg = gray.copy().astype("float")
			continue
		motions = motion(gray)

		'''
		for (left, top, right, bottom) in motions:
			cv2.rectangle(image_for_result, (left, top), (right, bottom), (255, 0, 0), 2)
			image = frame[top*PREPROCESS_DIMS:bottom*PREPROCESS_DIMS, left*PREPROCESS_DIMS:right*PREPROCESS_DIMS]
			cv2.imwrite('results/' + 'detect_' + str(time.time()) + '.jpg', image)
			predictions = predict(image)
			for i, (_left, _top, _right, _bottom) in enumerate(predictions):
				print ("detect {}/{} faces".format(i+1, len(predictions)))
				cv2.imwrite('results/' + 'face_' + str(time.time()) + '.jpg', image[_top:_bottom, _left:_right])
		cv2.imshow("Output", frame)
		key = cv2.waitKey(1) & 0xFF
		'''
		for (left, top, right, bottom) in motions:
			#zoom in the motion result
			left *= PREPROCESS_DIMS
			top *= PREPROCESS_DIMS
			right *= PREPROCESS_DIMS
			bottom *=PREPROCESS_DIMS
			#draw motions detect area in red
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
			image = frame[top : bottom, left : right]
			predictions = predict(image)
			# loop over our predictions
			for i, (_left, _top, _right, _bottom) in enumerate(predictions):
				#store the face detected
				cv2.imwrite('results/' + 'face_' + str(time.time()) + '.jpg', image[_top:_bottom, _left:_right])
				#draw faces in blue
				cv2.rectangle(frame, (left + _left, top + _top), (left + _right, top + _bottom), (255, 0, 0), 2)
				shape = sp(image, dlib.rectangle(_left, _top, _right, _bottom))
				face_descriptor = facerec.compute_face_descriptor(image, shape)
				matched = find_known_face(KNOWN_FACE_DICT, face_descriptor)
				print matched
				for i in range(len(matched)):
					cv2.putText(frame, "{}:{:.2f}".format(matched[i][0][0:2], 1 - matched[i][1]),
								(left , top + _top +  i * 15), cv2.FONT_HERSHEY_DUPLEX, 1,
								(0, 255, 0), 1)
					if i > 0:
						#list the top 2 most likely canditates
						break
			if len(predictions) > 0:
				cv2.imwrite('results/' + 'frame' + str(time.time()) + '.jpg', frame)
		# check if we should display the frame on the screen
		# with prediction data (you can achieve faster FPS if you
		# do not output to the screen)
		if args["display"] > 0:
			# display the frame to the screen
			fps.stop()
			cv2.putText(frame, "FPS = {:.2f}".format(fps.fps()), (100,100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
			cv2.imshow("Output", frame)
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
			if key == ord("s"):
				cv2.imwrite('results/' + 'frame' + str(time.time()) + '.jpg', frame)
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