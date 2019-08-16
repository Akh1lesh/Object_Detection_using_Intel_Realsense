
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pyrealsense2 as rs
from imutils.video import FPS
from imutils.video import VideoStream



ap = argparse.ArgumentParser()
# construct the argument parse and parse the arguments
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor","hand"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
fps = FPS().start()
try:

	while True:
		

		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		if not  color_frame:
			continue
		color_image = np.asanyarray(color_frame.get_data())
		color_image = imutils.resize(color_image,width=1000)
		(h, w) = color_image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)),
        0.007843, (300, 300), 127.5)
		net.setInput(blob)
		detections = net.forward()

	
		for i in np.arange(0, detections.shape[2]):
		# loop over each of the detections
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
			
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
				cv2.rectangle(color_image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(color_image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', color_image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
				break
	print("[INFO] cleaning up...")
	fps.stop()
finally: 
	pipeline.stop()
cv2.destroyAllWindows()
vs.stop()
