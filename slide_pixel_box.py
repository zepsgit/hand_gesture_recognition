from traceback import print_tb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# with open("./dataset/test_pixels.csv") as file_name:
#     test_data = np.loadtxt(file_name, delimiter=",")
with open("./dataset/pixels_v2.csv") as file_name:
    dataset = np.loadtxt(file_name, delimiter=",")
np.random.shuffle(dataset)
print(dataset.shape)
data=dataset[:15000]
test_data=dataset[15000:]
h, w = 48,68

############################
X_test=test_data[:,0:w-1]
y_test=test_data[:,w-1]
cnt = 0
#neigh = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(max_features=0.5)
# X_train = data[:, 0:w-1]
# y_train = data[:, w-1]
# neigh.fit(X_train, y_train)
# p_base = neigh.score(X_test, y_test)
# print(p_base)
for step in [1]:
    h_step=w_step=step
    diff=[]
    for i in range(0, h*w, w_step):
        print(cnt)
        cnt += 1
        X_train = data[:, 0:w-1]
        y_train = data[:, w-1]
        for j in range(h_step):
            X_train[i+j*w:i+j*w+w_step] = 0
        # X_train, X_test, y_train, y_test = train_test_split(X,
        #                                                     y,
        #                                                     random_state=42)
        rf.fit(X_train, y_train)
        p = rf.score(X_test, y_test)
        diff.append(p)
        np.savetxt('./dataset/slide_diff_rf_{m}x{n}'.format(m=step,n=step), diff)

# for step in range(1,2,1):
#     X_train = data[:, 0:w-1]
#     y_train = data[:, w-1]
#     diff=[]
#     for i in range(int(h/step-1)+1):
#         for j in range(i*step*w,(i*step+1)*w,step):
#             for k in range(step):
#                 X_train[j+k*w:j+k*w+step]=0
#             cnt+=1
#             print(cnt)
#             neigh.fit(X_train,y_train)
#             p=neigh.score(X_test,y_test)
#             diff.append(p_base-p)
#             np.savetxt('./dataset/slide_diff_v2_{m}x{n}'.format(m=step,n=step), diff)

