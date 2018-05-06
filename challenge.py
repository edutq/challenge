
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# calculates the eyebrow aspect ratio based on the positioon of the
# tip of the nose, the inner edge of the eyebrow and the center of 
# the eyebrow
def eyebrows_tip_of_nose(edge_of_brow, middle_point_brow, tip_of_nose):

	A = dist.euclidean(edge_of_brow, tip_of_nose)
	B = dist.euclidean(middle_point_brow, tip_of_nose)
	res = A/B
	return res

# give the status of the eyebrow based on the eyebrow aspect ratio
# value
def BAR (brow_aspect_ratio):

	if brow_aspect_ratio > 0.47:
		return "raised"

	elif brow_aspect_ratio <= 0.47 and  brow_aspect_ratio >= 0.40:
		return "normal"

	else:
		return "lowered"

# construct the argument parse 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")

args = vars(ap.parse_args())
 
# variables to display the status of each eyebrow
left_eyebrow_status = ""
right_eyebrow_status = ""

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor which is pre-trained

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# get the landmarks for each eyebrow and the nose
(reyebrowS, reyebrowE) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(leyebrowS, leyebrowE) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

(noseStart, noseEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

# start the camera video stream thread
vs = VideoStream(src=0).start()
# wait a second to make sure the camara is ready
time.sleep(1.0)

# loop over frames from the video stream
while True:

	# grab the frame from the threaded video stream, resize
	# it, and convert it to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	faces = detector(gray, 0)

	# loop over the face detections
	for face in faces:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)

		# get the left to right values of the eyebrows
		# top to bottom for the nose
		rightEyeBrow = shape[reyebrowS:reyebrowE]
		leftEyeBrow = shape[leyebrowS:leyebrowE]
		nose = shape[noseStart:noseEnd]

		# calculate the eyebrow aspect ratio of each brow
		REB_distance = eyebrows_tip_of_nose(rightEyeBrow[4], rightEyeBrow[2], nose[0])
		LEB_distance = eyebrows_tip_of_nose(leftEyeBrow[0], leftEyeBrow[2], nose[0])
		
		# use openCV convex hull to visualize the brows and the nose
		rightEyeBrowHull = cv2.convexHull(rightEyeBrow)
		leftEyeBrowHull = cv2.convexHull(leftEyeBrow)
		noseHull = cv2.convexHull(nose)

		cv2.drawContours(frame, [rightEyeBrowHull], -1, (0,255,0), 1)
		cv2.drawContours(frame, [leftEyeBrowHull], -1, (0,255,0), 1)
		cv2.drawContours(frame, [noseHull], -1, (0,255,0), 1)

		# get the eyebrows status
		right_eyebrow_status = BAR(REB_distance)
		left_eyebrow_status = BAR(LEB_distance)

		# draw the status of the eyebrows and the aspect ratio value 
		cv2.putText(frame, "L EyeBrow: {}".format(left_eyebrow_status), (10, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "R EyeBrow: {}".format(right_eyebrow_status), (10, 330),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "R EBAR: {:.2f}".format(REB_distance), (300, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "L EBAR: {:.2f}".format(REB_distance), (300, 330),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()