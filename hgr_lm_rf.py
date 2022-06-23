# TechVidvan hand Gesture Recognizer

# import necessary packages

import webbrowser
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import os 
from model_train import lm_rf
from model_train import lm_knn

class Flag:
  flag = True
flag1=Flag()
flag2=Flag()
flag3=Flag()
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

label_dic={
    0:'fist',
    1:'palm',
    2:'one',
    3:"ok",
    4:"thumb"
}
# Initialize the webcam
cap = cv2.VideoCapture(0)
cnt=1
while True:
    print("recognizing {} frames".format(cnt))
    cnt+=1
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append(lmx)
                landmarks.append(lmy)
        
            # Drawing landmarks on frames
        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
        prediction = lm_rf.predict([landmarks])[0]
        className = label_dic[prediction]

    if className=='one' and flag1.flag:
        webbrowser.open_new("www.google.com")
        setattr(flag1, 'flag', False)
    if className=='fist' and flag2.flag:
        webbrowser.open_new("www.bing.com")
        setattr(flag2, 'flag', False)

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()