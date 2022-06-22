import cv2 as cv
import numpy as np
import csv

pixels=[]
cnt=0
for name in ['fist','palm','ok','one','thumb']:
    for i in range(2000,8000):
        cnt+=1
        print(cnt)
        img = cv.imread('./train_img/{name}_{i}.jpg'.format(name=name,i=i),0)
        #resize original image
        scale_percent = 10 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        # cv.imshow("out",resized_img)
        # print(resized_img.shape)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        resized_img=resized_img.flatten()
        n=resized_img.shape[0]
        if name=='fist':
            resized_img=np.insert(resized_img,n,0)
        if name=="palm":
            resized_img=np.insert(resized_img,n,1)
        if name=="one":
            resized_img=np.insert(resized_img,n,2)
        if name=="ok":
            resized_img=np.insert(resized_img,n,3)
        if name=="thumb":
            resized_img=np.insert(resized_img,n,4)
        pixels.append(resized_img)
#print(pixels[0])
filename = './dataset/pixels_v2.csv'
with open(filename, 'w', newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerows(pixels)