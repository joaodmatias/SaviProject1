import cv2 as cv
import numpy as np
from copy import deepcopy
import cv2
from functions import Detection, Tracker
import face_recognition


#Import train
matias = face_recognition.load_image_file(f'faces\Matias.jpg')
matias_encoding = face_recognition.face_encodings(matias)[0]

vicente = face_recognition.load_image_file(f"faces\Vicente.jpg")
vicente_encoding = face_recognition.face_encodings(vicente)[0]

our_faces = [matias_encoding, vicente_encoding]
our_names = ['Matias', 'Vicente']

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

frame_counter = 0
while True:
    #Get the frame
    ret, frame_rgb = cap.read()
    frame = cv.flip(frame_rgb, 1)

    frame_counter +=1

    #Convert form bgr to rgb
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image_gui = deepcopy(frame)

    if ret == False:
        break
    stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000


    # ------------------------------------------
    # Detection of persons 
    # ------------------------------------------
    bboxes = face_recognition.face_locations(image_rgb)

    # ------------------------------------------
    # Create Detections per haar cascade bbox
    # ------------------------------------------
    detections = []
    for bbox in bboxes: 
        y1, x2, y2, x1 = bbox
        face_encoding = face_recognition.face_encodings(image_rgb, bbox)
        detection = Detection(y1, x2, y2, x1, image_rgb, id=detection_counter, stamp=stamp, face_encoding=face_encoding, our_faces=our_faces, our_names=our_names)
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
        




