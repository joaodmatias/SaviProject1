#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from copy import deepcopy
import cv2
from functions import Detection, Tracker
import face_recognition
import os


    #---------------------------------
    # Inititalization
    #---------------------------------
    
path = "/home/matias/Desktop/SAVI/TP1/SaviProject1/faces"
our_names = []
our_faces = []
for image in os.listdir(path):
    face_image = face_recognition.load_image_file(path + f'/{image}')
    face_encoding = face_recognition.face_encodings(face_image)[0]
    name, extention = image.split('.')
    our_names.append(name)
    our_faces.append(face_encoding)


face_locations = []
face_encodings = []
detection_counter = 0
tracker_counter = 0
trackers = []
iou_threshold = 0.8
names = []
frame_counter = 0

    #---------------------------------
    # Execution
    #---------------------------------
    
cap = cv.VideoCapture(0)

while True:
    # Get the frame
    ret, frame_rgb = cap.read()
    frame = cv.flip(frame_rgb, 1)

    frame_counter +=1

    # Convert form bgr to rgb for the face recognition library
    image_rgb = frame[:, :, ::-1]
    image_gui = deepcopy(frame)

    if ret == False:
        break

    # Get the time stamp
    stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000


    # face_recognition tool usage to find a face 
    face_locations = face_recognition.face_locations(image_rgb)
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
   

    # Creates a Detection class and connects it with a face
    detections = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        w = right-left
        h = bottom-top
        x1 = left
        y1 = top
        detection = Detection(x1, y1, w, h, image_rgb, id=detection_counter, stamp=stamp, face_encoding=face_encoding, our_faces=our_faces, our_names=our_names)
        detection_counter += 1
        detection.draw(image_gui)
        detections.append(detection)

 
    # Detection to tracker evaluation and association
    for detection in detections: 
        for tracker in trackers: 
            if tracker.active:
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)

                if iou > iou_threshold:  
                    tracker.addDetection(detection, image_rgb)

    # Track using template matching
    for tracker in trackers:
        last_detection_id = tracker.detections[-1].id
        detection_ids = [d.id for d in detections]
        if not last_detection_id in detection_ids:
            tracker.track(image_rgb)

    # Deactivate Tracker if it doesn't detect for two seconds
    for tracker in trackers: 
        tracker.updateTime(stamp)

    # Create a Tracker class for each detection
    for detection in detections:
        if not detection.assigned_to_tracker:
            tracker = Tracker(detection, id=tracker_counter, image=image_rgb, person = detection.person)
            tracker_counter += 1
            trackers.append(tracker)

    # ------------------------------------------
    # Draw stuff
    # ------------------------------------------

    # Draw trackers
    for tracker in trackers:
        if tracker.active:
            tracker.draw(image_gui)


    # Show image in window    
    cv2.imshow('Facial Recognizer 3000',image_gui) # show the image

    if cv2.waitKey(50) == ord('q'):
        break

    frame_counter += 1


# ------------------------------------------
# Termination
# ------------------------------------------
cap.release()
cv2.destroyAllWindows()
        



