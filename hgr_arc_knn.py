# Initialize the webcam
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model_train import arc_rf
from model_train import arc_knn
def resize_img(img):
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

def find_contour(img_arr,th):
    h,w=img_arr.shape
    for i in range(h):
        for j in range(w):
            if img_arr[i,j]>=th:
                img_arr[i,j]=0
            else:
                img_arr[i,j]=255

def edge_array(edged_img):
    contour = []
    h, w = edged_img.shape
    for m in range(h):
        for n in range(w):
            if edged_img[m, n] ==0:#110 better
                contour.append(m+1j*n)
    n=len(contour)
    if n<1518:
        while 1518-n>0:
            contour.append(0)
            n+=1
    else:
        contour=contour[0:1518]
    return np.array(contour)

def arc_len(edge_arr):
    d = np.round(np.abs(edge_arr))
    arc = np.angle(edge_arr)
    return np.array([np.round(d*arc)])

def rgb2gray(img):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_image = np.dot(img[...,:3], rgb_weights).astype(np.uint8)
    return grayscale_image


cap = cv2.VideoCapture(0)
cnt=1
print("mode training done")
label_dic={
    0:'fist',
    1:'palm',
    2:'one',
    3:"ok",
    4:"thumb"
}
while True:
    print("recognizing {} frames".format(cnt))
    cnt+=1
    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, c = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    framegray=rgb2gray(framergb)
    img=resize_img(framegray)
    canny=cv2.Canny(framegray,110,125)
    find_contour(img,110)
    X=edge_array(img)
    X_arc=arc_len(X)
    prediction=arc_knn.predict(X_arc)[0]
    #print(prediction[0])
    result=label_dic[prediction]
    # if cv2.waitKey(1)==ord('c'):#c for canny
    #     cv2.imshow("canny edge", canny) 
    # if cv2.waitKey(1)==ord('b'):# b for black
    #     cv2.imshow("gray",framegray)
    cv2.putText(canny, result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("rgb",canny)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
