from asyncio.constants import SSL_HANDSHAKE_TIMEOUT
import cv2 as cv
from cv2 import COLOR_BGR2GRAY
import numpy as np
from turtle import color
from copy import deepcopy
import face_recognition

haar_cascade = cv.CascadeClassifier('Trabalho1\haar_face.xml')

joao = cv.imread('Trabalho1\joao.jpg')
joao_gray = cv.cvtColor(joao, cv.COLOR_BGR2RGB)
joao_rect = haar_cascade.detectMultiScale(joao_gray, scaleFactor=1.1, minNeighbors=4, minSize=(150,150))
#x, y, w, h = joao_rect
#joao_roi = joao_gray[y:y+h, x:x+w]
for (x,y,w,h) in joao_rect:
    joao_roi = joao_gray[y:y+h, x:x+w]
joao_enc = face_recognition.face_encodings(joao_roi)

capture = cv.VideoCapture(0)

bbox_last_frame = []

count = 0
track_id = 0
tracking_ppl = {}

while True:
    ret, frame = capture.read()
    count +=1
    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    bbox = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(40,40))
    image_gui = deepcopy(frame)
    bbox_cur_frame = [] 

    for (x,y,w,h) in bbox:    #Put all the bboxes's coordinates found on the frame in an array and design a box arround it
        bbox_cur_frame.append((x,y,w,h))
        face_roi = gray[y:y+w, x:x+h]
        face_enc = face_recognition.face_encodings(face_roi)
        cv.rectangle(image_gui, (x,y), (x+w,y+h), (0,0,255), 2)
        result = face_recognition.compare_faces(joao_enc, face_enc)
        if result:
            cv.rectangle(image_gui, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            cv.rectangle(image_gui, (x,y), (x+w,y+h), (0,0,255), 2)


    if count <= 2:
        # Compare the area that is coincident with the bbox from last frame and the current frame
        for b1 in bbox_cur_frame:
            for b2 in bbox_last_frame:
                x1_intr = min(b1[0], b2[0])             
                y1_intr = min(b1[1], b2[1])             
                x2_intr = max(b1[0] + b1[2], b2[0] + b2[2]) 
                y2_intr = max(b1[1] + b1[3], b2[1] + b2[3])

                w_intr = x2_intr - x1_intr
                h_intr = y2_intr - y1_intr
                A_intr = w_intr * h_intr

                A_union = (b1[2] * b1[3]) + (b2[2] * b2[3]) - A_intr

                iou = A_intr / A_union

                if iou > 0.8:  #check if the diference is less than 20%
                    tracking_ppl[track_id] = b1  #add an id to a specific box
                    track_id += 1
    else:
        tracking_ppl_copy = tracking_ppl.copy()
        bbox_cur_frame_copy = bbox_cur_frame.copy()

        for ppl_id, b2 in tracking_ppl_copy.items():  #check the already exixtent boxes by id
            ppl_exists = False
            for b1 in bbox_cur_frame_copy:    #Compare the areas again
                x1_intr = min(b1[0], b2[0])             
                y1_intr = min(b1[1], b2[1])             
                x2_intr = max(b1[0] + b1[2], b2[0] + b2[2]) 
                y2_intr = max(b1[1] + b1[3], b2[1] + b2[3])

                w_intr = x2_intr - x1_intr
                h_intr = y2_intr - y1_intr
                A_intr = w_intr * h_intr

                A_union = (b1[2] * b1[3]) + (b2[2] * b2[3]) - A_intr

                iou = A_intr / A_union

                if iou > 0.8:
                    tracking_ppl[ppl_id] = b1   #update the bbox with the same id
                    ppl_exists = True
                    if b1 in bbox_cur_frame:
                        bbox_cur_frame.remove(b1)
                    continue

            if not ppl_exists:
                tracking_ppl.pop(ppl_id)
        
        for b in bbox_cur_frame:
            tracking_ppl[track_id] = b
            track_id += 1
                


    for ppl_id, b in tracking_ppl.items():
        cv.putText(image_gui, str(ppl_id), (b[0], b[1]), 0, 1, (0,0,255), 2)

    stamp = float(capture.get(cv.CAP_PROP_POS_MSEC))/1000









    # for (x,y,w,h) in face_rect:
    #     face_roi = gray[y:y+w, x:x+h]
    #     aa = cv.resize(joao_roi,w,h)
    #     errorL2 = cv.norm( face_roi, joao_roi, cv.NORM_L2 )
    #     similarity = 1 - errorL2 / ( h * w )
    #     if similarity > 0.7:
    #         cv.rectangle(image_gui, (x,y), (x+w,y+h), (0,255,0), 2)
    #     else:
    #         cv.rectangle(image_gui, (x,y), (x+w,y+h), (0,0,255), 2)


    bbox_last_frame = bbox_cur_frame.copy()
        
    cv.imshow('camara', image_gui)

    if cv.waitKey(20) & 0xFF ==ord('g'):
        break


capture.release()
cv.destroyAllWindows