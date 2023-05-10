# ASL Alphabet Recognition using TensorFlow CNN Model

This project aims to recognize the American Sign Language (ASL) alphabet using a convolutional neural network (CNN) model built with TensorFlow. The recognition of these gestures can help people with hearing and speech disabilities to communicate more effectively.

### REPORT
- [Pdf](./Main/Report.pdf)
- [Latex](./Main/Latex)
### Presentation
- [Pdf]()
### Video
---
## Dataset

The dataset used for this project is the ASL Alphabet Dataset from Kaggle, which can be downloaded from https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download

The dataset contains approximately 87,000 images of the ASL alphabet, each with a corresponding label indicating the letter represented in the image. The images are 200x200 pixels, and there are 29 classes in total, with 26 classes for the letters A-Z and 3 classes for SPACE, DELETE, and NOTHING. The inclusion of these three classes is particularly helpful for real-time applications and classification. The model can use the DELETE and SPACE classes to delete a letter and add a space, respectively, while the NOTHING class can help the model recognize when no gesture is being made.

Incase of downloading data from Kaggle:
```markdown
1. Rename data directory to "Data"
2. Rename both inner directory in "Data" to 1 and 2
3. Move sub-directories in "1" and "2" up into "Data"
4. Delete directory "1" and "2"
```
---
## Model Architecture

The model architecture used for this project is a CNN with **3 convolutional layers**, **each followed by a max-pooling layer and a batch normalization layer**. The **output of the final max-pooling layer** is flattened and fed into two dense layers, with the **final dense layer having 29 units** (one for each class). The activation function used in all layers is ReLU, except for the final dense layer, which uses softmax. The model was trained for **15 epochs** using the **Adam optimizer**, with a **batch size of 32** and a **learning rate of 0.001**.

---
## Results

The trained model achieved an accuracy of approximately **99.77% on the test set**. The model can correctly recognize hand gestures corresponding to the ASL alphabet, as well as the SPACE, DELETE, and NOTHING classes. The model can be further improved by using data augmentation techniques, such as rotation, translation, and scaling of the input images.

---
## Usage

To use the model, first install the required packages by running the following command in the terminal:

```python
pip install -r requirements.txt
```

After installing the required packages, you can run the App by running the following commanda in terminal:
```bash
cd Main
python App.py
```

Or, you can load the trained model from the saved model file using the following code:

```python
from tensorflow.keras.models import load_model

model = load_model('Saved_models/Model_sm')
```
To predict the class of an input image, you can pass the image to the model's predict() method as follows:

```python
import cv2
import numpy as np

# Load input image
img = cv2.imread('input_image.jpg')
Mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

# Pre Process
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #make img greyscale
img = cv2.resize(img, (32,32))                #resize image 

np_X = np.array(img)                          #Convert to numpy array
normalised_X = np_X.astype('float32')/255.0   #normalize numpy array
normalised_X=np.expand_dims(normalised_X,0)   #add an extra dimension b/c model is trained on a (None,32,32) shape

pred = model.predict(normalised_X,verbose=0)  #predict using model 
pred_index = np.argmax(pred)                  #index with max prediction

char = Mapping[pred_index]                    #map index to character
accuracy = pred[0][pred_index]*100            #accuracy/confidece of prediction

```
---
## PRETRAINED MODELS
- [HDF5](./Saved_models/Model_hdf5)
- [Tensorflow saved model](./Saved_models/Model_sm)
- [Tensorflow Lite](./Saved_models/Model_tflite.tflite)