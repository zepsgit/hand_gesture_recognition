# from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import label_binarize

# from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold
# with open("./dataset/dataset2.csv") as file_name:
#     train_data = np.loadtxt(file_name, delimiter=",")
# h_tr,w_tr=train_data.shape
# X_train=train_data[:,0:w_tr-1]
# y_train=train_data[:,w_tr-1]
# y_train = label_binarize(y_train, classes=[*range(5)])

# with open("./dataset/dataset1.csv") as file_name:
#     test_data = np.loadtxt(file_name, delimiter=",")
# h_ts,w_ts=test_data.shape
# X_test=test_data[:,0:w_ts-1]
# y_test=test_data[:,w_ts-1]
# y_test = label_binarize(y_test, classes=[*range(5)])

with open("./dataset/dataset12.csv") as file_name:
    data = np.loadtxt(file_name, delimiter=",")
h,w=data.shape
X=data[:,0:w-1]
y=data[:,w-1]
Y = label_binarize(y, classes=[*range(5)])

# neigh = KNeighborsClassifier(n_neighbors=7)
# kf = KFold(n_splits=5,shuffle=True)
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# neigh.fit(X_train, y_train)
# p=neigh.score(X_test,y_test)
# print(p)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics
#%matplotlib inline
label_dic={
    0:'fist',
    1:'palm',
    2:'one',
    3:"ok",
    4:"thumb"
}

X_train, X_test, y_train, y_test = train_test_split(X,
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
plt.savefig('./img/pr_10k12.png')
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
plt.savefig('./img/roc_10k12.png')########### only class 1 and class 2
plt.show() 
    
