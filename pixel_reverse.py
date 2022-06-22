import cv2 as cv
import numpy as np
import csv
from matplotlib import pyplot as plt
pixels=[]
cnt=0
for name in ['fist']:#,'palm','ok','one','thumb'
    for i in range(1):
        cnt+=1
        print(cnt)
        img = cv.imread('./train_img/{name}_{i}.jpg'.format(name=name,i=i),0)
        #resize original image
        scale_percent = 10 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)#(64,48)
        resized_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        # cv.imshow("out",resized_img)
        # print(resized_img.shape)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        resized_img=resized_img.flatten()
        # resized_img[1318:1321]=0
        # resized_img[1357:1369]=0
        # resized_img[1376:1380]=0
        # resized_img[1998:2000]=0
        # resized_img[2996:2998]=0
        resized_img[0:995]=0
        resized_img=np.reshape(resized_img,(-1,width))
        plt.imshow(resized_img, interpolation='nearest')
        plt.show()
        #     plt.show()
        #     plt.imshow(resized_img, interpolation='nearest')
        #     plt.show()
        # for j in range(20):
        #     resized_img[0:j]=0
        #     resized_img=np.reshape(resized_img,(-1,width))
        #     plt.imshow(resized_img, interpolation='nearest')
        #     plt.show()
