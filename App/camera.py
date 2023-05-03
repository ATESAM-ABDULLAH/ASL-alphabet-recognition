# import the opencv library
import cv2
  
  
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
    frame=cv2.rectangle(frame,top_left,bottom_right,255,2) #(image,top_left_corner,bottom_right_coroner,color,thickness)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Display the crop frame
    cv2.imshow('crop frame', cropframe)
      
    #Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the capture object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()