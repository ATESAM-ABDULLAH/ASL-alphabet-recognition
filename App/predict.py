# imports
import cv2
import numpy as np
from tensorflow import keras

# Constants
Mapping = {0: 'R', 1: 'Z', 2: 'nothing', 3: 'X', 4: 'D', 5: 'I', 6: 'E', 7: 'del', 8: 'O'
           , 9: 'K', 10: 'Q', 11: 'B', 12: 'G', 13: 'J', 14: 'Y', 15: 'V', 16: 'M', 17: 'P'
           , 18: 'A', 19: 'space', 20: 'H', 21: 'F', 22: 'T', 23: 'W', 24: 'S', 25: 'U', 26: 'C'
           , 27: 'L', 28: 'N'}
model_dir = "Saved_models/Modelv1_sm"
model = keras.models.load_model(model_dir)

# Predict func
def predict_img(img:'png'):
    img = cv2.resize(img, (32,32))                #resize image 

    np_X = np.array(img)                          #Convert to numpy array
    normalised_X = np_X.astype('float32')/255.0   #normalize numpy array
    normalised_X=np.expand_dims(normalised_X,0)   #add an extra dimension b/c model is trained on a (None,32,32,3) shape
    
    pred = model.predict(normalised_X,verbose=0)
    pred_index = np.argmax(pred)

    char = Mapping[pred_index]
    accuracy = pred[0][pred_index]*100

    return  char, accuracy

# Predict
img_dir = "Data/asl_alphabet_test/A_test.jpg"

img = cv2.imread(img_dir)   #read image in folder->png image

a,b = predict_img(img)      # Predict img

print(f"\nChar:{a} ,Accuracy:{b}% ")