#!/usr/bin/env python3
from copy import deepcopy
import numpy  as np
import cv2 as cv
import face_recognition

from functions import Detection

def main():

    #---------------------------------
    # Inititalization
    #---------------------------------
    
    cap = cv.VideoCapture(0)  #open camera video     
    if(cap.isOpened()==False):                 
        print('Error loading media')
    
    width_height = (800,600)
    
    #Recognizes face templates (do this for adding a person)
    matias = face_recognition.load_image_file("Matias.jpg")
    matias_encoding = face_recognition.face_encodings(matias)[0]
    
    vicente = face_recognition.load_image_file("Vicente.jpg")
    vicente_encoding = face_recognition.face_encodings(vicente)[0]
    
    our_faces = [matias_encoding, vicente_encoding]
    our_names = ['Matias', 'Vicente']
    
    #Initialize lists
    face_locations = []
    face_encodings = []
    names = []
    process_this_frame = True


    #---------------------------------
    # Execution
    #---------------------------------


    while(cap.isOpened()==True):

        ret, cam_vid = cap.read()    # Capture the video frame by frame
        vid_resized = cv.resize(cam_vid, width_height, interpolation= cv.INTER_LINEAR)

        if ret == False:
            break

        # mirror the video
        vid = cv.flip(vid_resized, 1)               

        vid_gui = deepcopy(vid)
        
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_vid = cv.resize(vid, (0, 0), fx=0.25, fy=0.25)


            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_vid = small_vid[:, :, ::-1]        
            
            #Searches faces in current frame
            face_locations = face_recognition.face_locations(rgb_small_vid)
            face_encodings = face_recognition.face_encodings(rgb_small_vid, face_locations)

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                found_face = face_recognition.compare_faces(our_faces, face_encoding)
                name = 'new name'
                
                face_distances = face_recognition.face_distance(our_faces, face_encoding)
                match_id = np.argmin(face_distances)
                if found_face[match_id]:
                    name = our_names[match_id]
                    print(name)

        
                    for (top, right, bottom, left) in face_locations:
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        # Draw a box around the face
                        cv.rectangle(vid_gui, (left, top), (right, bottom), (0, 0, 255), 2)

                        # Draw a label with a name below the face
                        cv.rectangle(vid_gui, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
                        font = cv.FONT_HERSHEY_DUPLEX
                        cv.putText(vid_gui, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                names.append(name)    
        
            cv.imshow('Camera video', vid_gui)       # Display the resulting frame
            if cv.waitKey(1) & 0xFF == ord('q'):     # the 'q' button is to quit
                break
        process_this_frame = not process_this_frame

      

    

    #---------------------------------
    # Termination
    #---------------------------------
    
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()