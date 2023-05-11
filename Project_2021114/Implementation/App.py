# import the opencv library
import cv2
import numpy as np
from tensorflow import keras

# Constants
Mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}
model_dir = "Model_sm"
model = keras.models.load_model(model_dir)

# Predict func
def predict_img(img:"numpy.ndarray")->"Char,Accuracy":
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #make img greyscale
    img = cv2.resize(img, (32,32))                #resize image 

    np_X = np.array(img)                          #Convert to numpy array
    normalised_X = np_X.astype('float32')/255.0   #normalize numpy array
    normalised_X=np.expand_dims(normalised_X,0)   #add an extra dimension b/c model is trained on a (None,32,32) shape
    
    pred = model.predict(normalised_X,verbose=0)  #predict using model 
    pred_index = np.argmax(pred)                  #index with max prediction

    char = Mapping[pred_index]                    #map index to character
    accuracy = pred[0][pred_index]*100            #accuracy/confidece of prediction

    return  char, accuracy

# Webcam capture function
def camera(top_left:list=[0,40],bottom_right:list=[300,340])->'Webcam':
    vid = cv2.VideoCapture(0)# define a video capture object

    while(True): #Loop over frames/images till closed
        # Capture the video frame by frame
        connected, frame = vid.read()
        #Connection not succesful
        if(not connected):
            break

        # Make a detection area -> model input
        cropframe=frame[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
        
        # Predict img
        char,accuracy = predict_img(cropframe)     
        
        # Draw Detection Area on frame
        frame=cv2.rectangle(frame,top_left,bottom_right,255,2) 

        # A solid background for Model output
        cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
        
        # Show Model output text
        cv2.putText(frame,f"Char: {char} - {round(accuracy,2)}% ", 
                    (3,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show message for exit
        cv2.putText(frame,"Press 'q' to quit", 
                    (170,460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the resulting frame
        cv2.imshow('Webcam', frame)        
        #Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the capture object
    vid.release()

    # Destroy all the windows
    cv2.destroyAllWindows()

camera()
