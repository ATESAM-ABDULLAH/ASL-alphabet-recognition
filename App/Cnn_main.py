# import the opencv library
import cv2
import numpy as np
from tensorflow import keras

# Constants
Mapping = {0: 'R', 1: 'Z', 2: 'nothing', 3: 'X', 4: 'D', 5: 'I', 6: 'E', 7: 'del', 8: 'O'
           , 9: 'K', 10: 'Q', 11: 'B', 12: 'G', 13: 'J', 14: 'Y', 15: 'V', 16: 'M', 17: 'P'
           , 18: 'A', 19: 'space', 20: 'H', 21: 'F', 22: 'T', 23: 'W', 24: 'S', 25: 'U', 26: 'C'
           , 27: 'L', 28: 'N'}
model_dir = "Saved_models/Cnn/Modelv1_sm"
model = keras.models.load_model(model_dir)

# Predict func
def predict_img(img:"numpy.ndarray")->"Char,Accuracy":
    img = cv2.resize(img, (32,32))                #resize image 

    np_X = np.array(img)                          #Convert to numpy array
    normalised_X = np_X.astype('float32')/255.0   #normalize numpy array
    normalised_X=np.expand_dims(normalised_X,0)   #add an extra dimension b/c model is trained on a (None,32,32,3) shape
    
    pred = model.predict(normalised_X,verbose=0)  #predict using model 
    pred_index = np.argmax(pred)                  #index with max prediction

    char = Mapping[pred_index]                    #map index to character
    accuracy = pred[0][pred_index]*100            #accuracy/confidece of prediction

    return  char, accuracy

# Webcam capture function
def camera(top_left:list=[0,40],bottom_right:list=[300,340])->'Webcam':
    vid = cv2.VideoCapture(0)# define a video capture object

    while(True): #Loop over frames/images till closed
        connected, frame = vid.read()# Capture the video frame by frame
        if(not connected):#Connection not succesful
            break

        # Make a detection area -> model input
        cropframe=frame[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
        # Make a rectangle on frame
        frame=cv2.rectangle(frame,top_left,bottom_right,255,2) 
        # Predict img
        char,accuracy = predict_img(cropframe)      # Predict img
        # A solid background for text
        cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
        # Show Model output
        cv2.putText(frame,f"Char: {char} - {round(accuracy,2)}% ", 
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

camera()