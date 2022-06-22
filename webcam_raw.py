import cv2
cap=cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    cv2.imshow("output",image)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
