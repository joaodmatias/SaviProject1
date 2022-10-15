#!/usr/bin/env python3
import numpy  as np
import cv2 as cv

def main():

    #---------------------------------
    # Inititalization
    #---------------------------------

    cap = cv.VideoCapture(0)  #open camera video
    if(cap.isOpened()==False):                 
        print('Error loading media')


    #---------------------------------
    # Execution
    #---------------------------------

    while(cap.isOpened()==True):

        ret, cam_vid = cap.read()                # Capture the video frame by frame
        
        vid = cv.flip(cam_vid, 1)                # mirror the video
    
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