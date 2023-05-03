# import the opencv library
import cv2
from predict import *
  
  
# define a video capture object
vid = cv2.VideoCapture(0)

# Detection Area points
top_left = [0,40]
bottom_right = [300,340]

while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    
    # Make a detection area -> model input
    cropframe=frame[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]

    # Make a rectangle on frame
    frame=cv2.rectangle(frame,top_left,bottom_right,255,2) 

    # Predict img
    char,accuracy = predict_img(cropframe)      # Predict img

    # A solid background for text
    cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)

    # Show Model output
    cv2.putText(frame,f"Char: {char} - {accuracy}% ", 
                (3,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)
      
    #Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the capture object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()