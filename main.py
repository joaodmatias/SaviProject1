#!/usr/bin/env python3
import numpy  as np
import cv2 as cv

def main():

    #---------------------------------
    # Inititalization
    #---------------------------------
    
    #uses a cascade function which will apply different "filters" one by one until it detects an actual face 
    #(algorithm already developed into opencv "Viola-Jones")
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml') 

    matias_color = cv.imread("../SaviProject1/Matias.jpg")    #template João Matias
    matias = cv.cvtColor(matias_color, cv.COLOR_BGR2GRAY) 


    cap = cv.VideoCapture(0)  #open camera video     
    if(cap.isOpened()==False):                 
        print('Error loading media')
   
    width  = int(cap.get(3)) 
    height = int(cap.get(4)) 


    matias_resized = cv.resize(matias, (720,640), interpolation = cv.INTER_AREA)
    h,w = matias_resized.shape
    cv.imshow('Matias', matias_resized)
    
    print("h= " + str(h) + " w= " + str(w) + " width= " + str(width) + " height= " + str(h))

    #---------------------------------
    # Execution
    #---------------------------------

    while(cap.isOpened()==True):

        ret, cam_vid = cap.read()    # Capture the video frame by frame
        
        vid = cv.flip(cam_vid, 1)                # mirror the video
        vid_gray = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)   #convert to gray

        #Detects objects of different sizes (faces in this case) in the video and returns them as list of rectangles.
        faces = face_cascade.detectMultiScale(vid_gray, scaleFactor=1.1, minNeighbors=4, minSize=(150,150))

    
        for (x, y, w, h) in faces:
            cv.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 2)   #draws rectangle in face
    

        match_matias = cv.matchTemplate(vid_gray, matias, cv.TM_CCOEFF_NORMED)   #matches template with video
        _, max_val, _, max_loc = cv.minMaxLoc(match_matias)


        if match_matias is True:
            print('got a match')

        cv.imshow('Camera video', vid)           # Display the resulting frame
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