import cv2 as cv
import numpy as np
from copy import deepcopy
import cv2
from functions import Detection, Tracker

#Definir haar cascade
haar_cascade = cv.CascadeClassifier(r'Trabalho1\haar_face.xml')

#Import train
people = ['Joao', 'Matias']
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'Trabalho1\face_trained.yml')

#essential variables
detection_counter = 0
tracker_counter = 0
trackers = []
iou_threshold = 0.8

#Execution
cap = cv.VideoCapture(0)

frame_counter = 0
while True:
    #Get the frame
    ret, frame_rgb = cap.read()
    frame = cv.flip(frame_rgb, 1)
    frame_counter +=1
    image_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    image_gui = deepcopy(frame)

    if ret == False:
        break
    stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000


    # ------------------------------------------
    # Detection of persons 
    # ------------------------------------------
    bboxes = haar_cascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=10, minSize=(100,100))

    # ------------------------------------------
    # Create Detections per haar cascade bbox
    # ------------------------------------------
    detections = []
    for bbox in bboxes: 
        x1, y1, w, h = bbox
        detection = Detection(x1, y1, w, h, image_gray, id=detection_counter, stamp=stamp, face_recognizer = face_recognizer, people = people)
        detection_counter += 1
        detection.draw(image_gui)
        detections.append(detection)
        # cv2.imshow('detection ' + str(detection.id), detection.image  )

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
                    tracker.addDetection(detection, image_gray)

    # ------------------------------------------
    # Track using template matching
    # ------------------------------------------
    for tracker in trackers: # cycle all trackers
        last_detection_id = tracker.detections[-1].id
        print(last_detection_id)
        detection_ids = [d.id for d in detections]
        if not last_detection_id in detection_ids:
            print('Tracker ' + str(tracker.id) + ' Doing some tracking')
            tracker.track(image_gray)

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
            tracker = Tracker(detection, id=tracker_counter, image=image_gray, person = detection.person)
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
        




