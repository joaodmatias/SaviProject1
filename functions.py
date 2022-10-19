import cv2 as cv
import numpy as np

class Box():

    def __init__(self, x1, y1, w, h):
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h

        self.area = w*h

        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h

    def IOU(self, bboxfinal):
    
        x1_intr = min(self.x1, bboxfinal.x1)             
        y1_intr = min(self.y1, bboxfinal.y1)             
        x2_intr = max(self.x2, bboxfinal.x2)
        y2_intr = max(self.y2, bboxfinal.y2)

        w_intr = x2_intr - x1_intr
        h_intr = y2_intr - y1_intr
        A_intr = w_intr * h_intr

        A_union = self.area + bboxfinal.area - A_intr
        
        return A_intr / A_union


class Detection(Box):
    
    def __init__(self, x1, y1, w, h, vid, counter):
        super().__init__(x1, y1, w, h)
        self.counter =  counter
        self.get_small_image(vid)

    def get_small_image(self, vid):
        self.vid = vid[self.y1+self.h, self.x1+self.w]

    def draw_detection(self, vid, color=(255,0,0)):
        cv.rectangle(vid,(self.x1,self.y1),(self.x2, self.y2),color,3)

        vid = cv.putText(vid, 'D' + str(self.counter), (self.x1, self.y1-5), cv.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv.LINE_AA)