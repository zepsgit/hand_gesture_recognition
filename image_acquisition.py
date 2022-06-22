import cv2
vidcap = cv2.VideoCapture(0)
count = 0
while vidcap.isOpened():
  success,image = vidcap.read()
  image = cv2.flip(image, 1)
  cv2.imwrite("./train_img/thumb_%d.jpg" % count, image)     # save frame as JPEG file      
  count += 1
  print('Read %d frame: ' % count,success)
  print(count)
  cv2.imshow("output",image)
  if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
vidcap.release()