from cProfile import label
import cv2
import mediapipe as mp
import tensorflow as tf
import csv

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
features=[]
labels=[]
cnt=0
for name in ['fist','palm','one','ok','thumb']:
    for i in range(10000):
        cnt+=1
        print(cnt)
        image = cv2.imread('./train_img/{name}_{i}.jpg'.format(name=name,i=i))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        x, y, c = image.shape
            # Get hand landmark prediction
        result = hands.process(image)
        print(image.shape)
        print(result.multi_hand_landmarks)
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
            if name=='fist':
                landmarks.append(0)
            if name=="palm":
                landmarks.append(1)
            if name=="one":
                landmarks.append(2)
            if name=="ok":
                landmarks.append(3)
            if name=="thumb":
                landmarks.append(4)
            features.append(landmarks)

filename = './dataset/dataset.csv'
with open(filename, 'w', newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerows(features)        


