#!/usr/bin/env python3
from copy import deepcopy
import numpy  as np
import cv2 as cv

from functions import Detection

def main():

    #---------------------------------
    # Inititalization
    #---------------------------------
    
    #uses a cascade function which will apply different "filters" one by one until it detects an actual face 
    #(algorithm already developed into opencv "Viola-Jones")

    matias_color = cv.imread("../SaviProject1/Matias.jpg")    #template João Matias
    width_height = (800, 600)
    matias_resized = cv.resize(matias_color, width_height, interpolation= cv.INTER_LINEAR)

    matias_copy = matias_resized.copy()
    matias = cv.cvtColor(matias_copy, cv.COLOR_BGR2GRAY) 

    tracker = cv.TrackerCSRT_create()

    cap = cv.VideoCapture(0)  #open camera video     
    if(cap.isOpened()==False):                 
        print('Error loading media')
   
    width  = int(cap.get(3)) 
    height = int(cap.get(4)) 

    window_name = 'Camera_video'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 500)

    
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml') 

    while(cap.isOpened()==True):

        ret, cam_vid = cap.read()    # Capture the video frame by frame
        vid_resized = cv.resize(cam_vid, width_height, interpolation= cv.INTER_LINEAR)

        if ret == False:
            break

        vid = cv.flip(vid_resized, 1)                # mirror the video
        
        vid_gray = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)   #convert to gray

        vid_gui = deepcopy(vid)

        #Detects objects of different sizes (faces in this case) in the video and returns them as list of rectangles.
        boxes = face_cascade.detectMultiScale(vid_gray, scaleFactor=1.1, minNeighbors=4, minSize=(150,150))


        timer = cv.getTickCount()
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

        for box in boxes:
            x1, y1, w, h = box
            ret = tracker.init(vid, box)
            ret, box = tracker.update(vid)
            if ret:
                p1 = (int(x1), int(y1))
                p2 = (int(x1 + w), int(y1 + h))
                cv.rectangle(vid_gui, p1, p2, (255,0,0), 2, 1)
        
            match_matias = cv.matchTemplate(vid_gray, matias, cv.TM_CCOEFF_NORMED)   #matches template with video
            _, max_val, _, max_loc = cv.minMaxLoc(match_matias)
            cv.rectangle(vid_gui, (max_loc[0], max_loc[1]), (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
        
            cv.putText(vid_gui, "Matias", 
                        (max_loc[0], max_loc[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 
                        2, lineType=cv.LINE_AA)

            print(str(x1) + ' ' +  str(y1) +' ' + str(max_loc[0]) +' ' + str(max_loc[1]))
        
        
       

        cv.imshow('Camera video', vid_gui)       # Display the resulting frame
        if cv.waitKey(1) & 0xFF == ord('q'):     # the 'q' button is to quit
            break
    

    #---------------------------------
    # Termination
    #---------------------------------
    
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv.destroyAllWindows()
    
        
    cv.waitKey(0)

if __name__ == "__main__":
    main()