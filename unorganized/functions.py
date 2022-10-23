#!/usr/bin/env python3

import csv
from copy import deepcopy
from turtle import color

import cv2
import numpy as np
from colorama import Fore, Style, Back
import face_recognition


class BoundingBox:
    
    def __init__(self, x1, y1, w, h):
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        self.area = w * h

        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h


    def computeIOU(self, bbox2):
    
        x1_intr = min(self.x1, bbox2.x1)             
        y1_intr = min(self.y1, bbox2.y1)             
        x2_intr = max(self.x2, bbox2.x2)
        y2_intr = max(self.y2, bbox2.y2)

        w_intr = x2_intr - x1_intr
        h_intr = y2_intr - y1_intr
        A_intr = w_intr * h_intr

        A_union = self.area + bbox2.area - A_intr
        
        return A_intr / A_union

    def extractSmallImage(self, image_full):
        return image_full[self.y1:self.y1+self.h, self.x1:self.x1+self.w]



class Detection(BoundingBox):

    def __init__(self, x1, y1, w, h, image_full, id, stamp, face_encoding, our_faces, our_names):
        super().__init__(x1,y1,w,h) # call the super class constructor        
        self.id = id
        self.stamp = stamp
        self.image =self.extractSmallImage(image_full)
        self.assigned_to_tracker = False
        # See if the face is a match for the known face(s)
        found_face = face_recognition.compare_faces(our_faces, face_encoding)
        self.person = 'new name'
        
        face_distances = face_recognition.face_distance(our_faces, face_encoding)
        match_id = np.argmin(face_distances)
        if found_face[match_id]:
            self.person = our_names[match_id]

    def draw(self, image_gui, color=(255,0,0)):
        cv2.rectangle(image_gui,(self.x1,self.y1),(self.x2, self.y2),color,3)

        image = cv2.putText(image_gui, 'd' , (self.x1, self.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)


class Tracker():

    def __init__(self, detection, id, image, person):
        self.id = id
        self.template = None
        self.active = True
        self.bboxes = []
        self.detections = []
        self.tracker = cv2.TrackerCSRT_create()
        self.time_since_last_detection = None
        self.person = person

        self.addDetection(detection, image)




    def getLastDetectionStamp(self):
        return self.detections[-1].stamp

    def updateTime(self, stamp):
        self.time_since_last_detection = round(stamp-self.getLastDetectionStamp(),1)

        if self.time_since_last_detection > 2: # deactivate tracker        
            self.active = False

    def drawLastDetection(self, image_gui, color=(255,0,255)):
        last_detection = self.detections[-1] # get the last detection

        cv2.rectangle(image_gui,(last_detection.x1,last_detection.y1),
                      (last_detection.x2, last_detection.y2),color,3)

        image = cv2.putText(image_gui, str(self.person) + ' T' + str(self.id), 
                            (last_detection.x2-40, last_detection.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

    def draw(self, image_gui, color=(255,0,255)):


        bbox = self.bboxes[-1] # get last bbox

        cv2.rectangle(image_gui,(bbox.x1,bbox.y1),(bbox.x2, bbox.y2),color,3)

        cv2.putText(image_gui, str(self.person) + ' T' + str(self.id), 
                            (bbox.x1+25, bbox.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

        cv2.putText(image_gui, str(self.time_since_last_detection) + ' s', 
                            (bbox.x1, bbox.y1-30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)


    def addDetection(self, detection, image):

        self.tracker.init(image, (detection.x1, detection.y1, detection.w, detection.h))

        self.detections.append(detection)
        detection.assigned_to_tracker = True
        self.template = detection.image
        bbox = BoundingBox(detection.x1, detection.y1, detection.w, detection.h)
        self.bboxes.append(bbox)

    def track(self, image):

        ret, bbox = self.tracker.update(image)
        x1,y1,w,h = bbox

#         h,w = self.template.shape
#         result = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
#         _, max_val, _, max_loc = cv2.minMaxLoc(result)
# 
#         x1 = max_loc[0] 
#         y1 = max_loc[1] 

        bbox = BoundingBox(x1, y1, w, h)
        self.bboxes.append(bbox)

        # Update template using new bbox coordinates
        self.template = bbox.extractSmallImage(image)
        
    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return text
