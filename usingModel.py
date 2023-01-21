import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import webbrowser
import pyautogui

#class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
classes = ["palm", "l"]
model = load_model('handrecognition_model.h5')
# Open Camera
capture = cv2.VideoCapture(0)

#setting resolution to 480p
#capture.set(3, 640)
#capture.set(4, 480)

while capture.isOpened():

    _, frame = capture.read()
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 1)
    crop_image = frame[100:300, 100:300]
    #crop_img = img[y:y + height, x:x + width]
    #frameCopy = frame.copy()
    frameCopy = cv2.resize(crop_image, (120, 320))
    gray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray)
    img_array = img_array.reshape(120, 320, 1)
    img_array = np.expand_dims(img_array, axis=0)

    #prediction = int(model.predict(img_array)[0][0])

    prediction = model.predict(img_array)
    #prediction[0,3:10] = 0
    prediction = prediction[0,1:3]
    value = np.argmax(prediction)
    #print("Prediction=", prediction)
    #print("Prediction=", type(prediction))
    #print("Label=", classes[value])
    #print("Value = ",value)

    #cv2.putText(frame,class_names[value], (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)


    #if class_names[value] == "l":
    if classes[value] == "l":
        pyautogui.press('space')
        cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

#Yes, this is correct the parameter input_shape is prepared to take 3 values. However the function Conv2D is expecting a 4D array as input, covering:

#Number of samples
#Number of channels
#Image width
#Image height
#Whereas the function load_data() is a 3D array consisting of width, height and number of samples.

#You can expect to solve the issue with a simple reshape:

#train_X = train_X.reshape(-1, 28,28, 1)
#test_X = test_X.reshape(-1, 28,28, 1)
#A better defitinion from keras documentation:

#Input shape: 4D tensor with shape: (batch, channels, rows, cols) if data_format is "channels_first" or 4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".

#https://towardsdatascience.com/tutorial-using-deep-learning-and-cnns-to-make-a-hand-gesture-recognition-model-371770b63a51

#https://github.com/filipefborba/HandRecognition