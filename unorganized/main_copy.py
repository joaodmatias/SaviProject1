#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from copy import deepcopy
import cv2
from functions import Detection, Tracker
import face_recognition
import os, sys

path = "/home/matias/Desktop/SAVI/TP1/SaviProject1/faces"
our_names = []
our_faces = []
string = 'jpg'
for image in os.listdir(path):
    face_image = face_recognition.load_image_file(path + f'/{image}')
    face_encoding = face_recognition.face_encodings(face_image)[0]
    if string in image:
        image.replace(string,'')
        our_names.append(image)
    our_faces.append(face_encoding)

#essential variables
face_locations = []
face_encodings = []
detection_counter = 0
tracker_counter = 0
trackers = []
iou_threshold = 0.8
names = []

#Execution
cap = cv.VideoCapture(0)

process_this_frame = True
frame_counter = 0

while (cap.isOpened()==True):
    #Get the frame
    ret, frame_rgb = cap.read()
    frame = cv.flip(frame_rgb, 1)

    frame_counter +=1

    #small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

    #Convert form bgr to rgb
    image_rgb = frame[:, :, ::-1]
    image_gui = deepcopy(frame)

    if ret == False:
        break
    stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000

    # ------------------------------------------
    # Detection of persons 
    # ------------------------------------------
    face_locations = face_recognition.face_locations(image_rgb)
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
    # ------------------------------------------
    # Create Detections per haar cascade bbox
    # ------------------------------------------

    detections = []
    first_time = True
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        w = right-left
        h = bottom-top
        x1 = left
        y1 = top

        detection = Detection(x1, y1, w, h, image_rgb, id=detection_counter, stamp=stamp, face_encoding=face_encoding, our_faces=our_faces, our_names=our_names, first_time=first_time)
        detection_counter += 1
        detection.draw(image_gui)
        detections.append(detection)
        # cv2.imshow('detection ' + str(detection.id), detection.image  )
        first_time = False
    # ------------------------------------------
    # For each detection, see if there is a tracker to which it should be associated
    # ------------------------------------------
    for detection in detections: # cycle all detections
        for tracker in trackers: # cycle all trackers
            if tracker.active:
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                # print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                if iou > iou_threshold: # associate detection with tracker 
                    tracker.addDetection(detection, image_rgb)

    # ------------------------------------------
    # Track using template matching
    # ------------------------------------------
    for tracker in trackers: # cycle all trackers
        last_detection_id = tracker.detections[-1].id
        #print(last_detection_id)
        detection_ids = [d.id for d in detections]
        if not last_detection_id in detection_ids:
            #print('Tracker ' + str(tracker.id) + ' Doing some tracking')
            tracker.track(image_rgb)

    # ------------------------------------------
    # Deactivate Tracker if no detection for more than T
    # ------------------------------------------
    for tracker in trackers: # cycle all trackers
        tracker.updateTime(stamp)

    # ------------------------------------------
    # Create Tracker for each detection
    # ------------------------------------------
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


        # win_name= 'T' + str(tracker.id) + ' template'
        # cv2.imshow(win_name, tracker.template)

    # for tracker in trackers:
        # print(tracker)

    cv2.imshow('window_name',image_gui) # show the image

    if cv2.waitKey(50) == ord('q'):
        break

    frame_counter += 1


# ------------------------------------------
# Termination
# ------------------------------------------
cap.release()
cv2.destroyAllWindows()
        



