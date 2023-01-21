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

while capture.isOpened():

    _, frame = capture.read()
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 1)
    crop_image = frame[100:300, 100:300]
    frameCopy = cv2.resize(crop_image, (120, 320))
    gray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray)
    img_array = img_array.reshape(120, 320, 1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prediction = prediction[0,1:3]
    value = np.argmax(prediction)

    if classes[value] == "l":
        pyautogui.press('space')
        cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
