from cgitb import grey
import csv
from filecmp import cmp
from cv2 import magnitude
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from csv import reader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import label_binarize
img = cv.imread('./train_img/fist_166.jpg', 0)
edges = cv.Canny(img, 125, 180)


def show_canny():
    for name in ['fist', 'palm', 'one', 'ok', 'thumb']:
        img = cv.imread('./train_img/{name}_166.jpg'.format(name=name), 0)
        edges = cv.Canny(img, 125, 180)
        h, w = edges.shape
        cv.imshow("canny", edges)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite('./img/canny_{name}.jpg'.format(name=name), edges)

def find_contour(img_arr,th):
    h,w=img_arr.shape
    for i in range(h):
        for j in range(w):
            if img_arr[i,j]>=th:
                img_arr[i,j]=0
            else:
                img_arr[i,j]=255
def show_pixel_contour():
    df = pd.read_csv('./dataset/test_pixels.csv')
    data=pd.DataFrame.to_numpy(df)
    for i in [0,1000,2000,3000,4000]:
        img=data[i][:3072]
        resized_img=np.reshape(img,(-1,64))
        find_contour(resized_img,115)
        plt.imshow(resized_img)
        plt.savefig('./img/pixel_contour_{}.jpg'.format(i))
        plt.show()

def edge_array(edged_img):
    contour = []
    h, w = edged_img.shape
    for m in range(h):
        for n in range(w):
            if edged_img[m, n] >=110:#110 better
                contour.append(m+1j*n)
    return np.array(contour)

h, w = edges.shape
# print("original image pixel", h*w)#307200
#edge_arr = edge_array(edges)  # 10939
# print(edge_arr)
# print("contour pixel", edge_arr.shape[0])

# calculate distance of contour coordinates as feature


def distance(edge_arr):
    return np.round(np.abs(edge_arr))


def calculate_digits(num):
    cnt = 0
    if num == 0:
        return 1
    while int(num):
        num /= 10
        cnt += 1
    return cnt


def catenate_twonum(number_a, number_b):
    nb = calculate_digits(number_b)
    return np.power(10, nb)*number_a+number_b


def catenate_arr(edge_arr):
    cat = []
    for i in edge_arr:
        x = catenate_twonum(np.real(i), np.imag(i))
        cat.append(x)
    return np.array(cat)


def angle_arr(edge_arr):
    return np.angle(edge_arr)


def arc_len(edge_arr):
    d = np.round(np.abs(edge_arr))
    arc = np.angle(edge_arr)
    return np.round(d*arc)


def get_all_contours():
    df = pd.read_csv('./dataset/pixels_v2.csv')
    data=pd.DataFrame.to_numpy(df)
    cnt=0
    contours=[]
    for i in range(data.shape[0]):
        print(cnt)
        cnt+=1
        img=data[i][:3072]
        resized_img=np.reshape(img,(-1,64))
        contour=edge_array(resized_img)
        n=contour.shape[0]
        if 0<=i<6000:
            contour=np.insert(contour,n,0)
        if 6000<=i<12000:
            contour=np.insert(contour,n,1)
        if 12000<=i<18000:
            contour=np.insert(contour,n,2)
        if 18000<=i<24000:
            contour=np.insert(contour,n,3)
        if 24000<=i<30000:
            contour=np.insert(contour,n,4)
        contours.append(contour)
    filename = './dataset/contours_coordinates.csv'
    with open(filename, 'w', newline='') as myfile:
        writer = csv.writer(myfile)
        writer.writerows(contours)

# get_all_contours()  

def read_contour():
    cnt=0
    # min lenth is 1518 ; max is 1847 of each contour
    with open('./dataset/contours_coordinates.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        X_arr=[]
        y_arr=[]
        for row in csv_reader:
            new_X=[]
            cnt+=1
            print(cnt)
            n=len(row)
            X=row[:1518]
            for x in X:
                new_X.append(complex(x))
            X_arr.append(new_X)
            y=complex(row[n-1])
            y_arr.append(y)
    return np.array(X_arr),np.array(y_arr)

def all_cat_arr(X_arr,y_arr):
    X_cats=[]
    n=X_arr.shape[0]
    for i in range(n):
        k=X_arr[i].shape[0]
        X_cat=catenate_arr(X_arr[i])
        X_cat=np.insert(X_cat,k,np.real(y_arr[i]))
        X_cats.append(X_cat)
    return np.array(X_cats)

def get_cat_dataset():
    X_arr,y_arr=read_contour()
    X_cats=all_cat_arr(X_arr,y_arr)

    filename_x = './dataset/catenate_dataset.csv'
    with open(filename_x, 'w', newline='') as myfile_x:
        writer = csv.writer(myfile_x)
        writer.writerows(X_cats)

def all_dis_arr(X_arr):
    X_dises=[]
    for x in X_arr:
        X_dis=distance(x)
        X_dises.append(X_dis)
    return np.array(X_dises)

def get_dis_dataset():
    X_arr,y_arr=read_contour()
    X_dises=all_dis_arr(X_arr)
    filename_x = './dataset/distance_dataset.csv'
    with open(filename_x, 'w', newline='') as myfile_x:
        writer = csv.writer(myfile_x)
        writer.writerows(X_dises)
#get_dis_dataset()
def all_arc_len(X_arr):
    X_arcs=[]
    for x in X_arr:
        X_arc=arc_len(x)
        X_arcs.append(X_arc)
    return np.array(X_arcs)
def get_arc_dataset():
    X_arr,y_arr=read_contour()
    X_arcs=all_arc_len(X_arr)
    filename_x = './dataset/arclen_dataset.csv'
    with open(filename_x, 'w', newline='') as myfile_x:
        writer = csv.writer(myfile_x)
        writer.writerows(X_arcs)
#get_arc_dataset()

############################
###### feature evaluation
###########################
with open("./dataset/catenate_dataset.csv") as file_name:
    cat_data = np.loadtxt(file_name, delimiter=",")

with open("./dataset/arclen_dataset.csv") as file_name:
    arc_data = np.loadtxt(file_name, delimiter=",")
with open("./dataset/distance_dataset.csv") as file_name:
    dis_data = np.loadtxt(file_name, delimiter=",")
h,w=cat_data.shape
X_cat=cat_data[:,0:w-1]
X_arc=arc_data[:,0:w-1]
X_dis=dis_data[:,0:w-1]

y=cat_data[:,w-1]
Y = label_binarize(y, classes=[*range(5)])

label_dic={
    0:'fist',
    1:'palm',
    2:'one',
    3:"ok",
    4:"thumb"
}

X_train, X_test, y_train, y_test = train_test_split(X_dis,
                                                    Y,
                                                    random_state = 42)

clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=50,
                             max_depth=3,
                             random_state=0))
clf.fit(X_train, y_train)

y_score = clf.predict_proba(X_test)

# precision recall curve
precision = dict()
recall = dict()
for i in range(5):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='{label} AP: {ap}'.format(label=label_dic[i],
                                    ap=np.around(np.average(precision[i]),2)))
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.savefig('./img/dis_feature_pr.png')
plt.show()
# roc curve
fpr = dict()
tpr = dict()

for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i],
                                  y_score[:, i])
    plt.plot(fpr[i], tpr[i], lw=2, label='{label}'.format(label=label_dic[i]))

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc="best")
plt.title("ROC curve")
plt.savefig('./img/dis_feature_roc.png')
plt.show() 
    
