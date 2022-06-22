# from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

with open("./dataset/dataset2.csv") as file_name:
    train_data_raw = np.loadtxt(file_name, delimiter=",")
_,w_train=train_data_raw.shape
train_data=[]
for row in train_data_raw:
    if row[w_train-1] in [0,1]:
        train_data.append(row)

with open("./dataset/dataset1.csv") as file_name:
    test_data_raw = np.loadtxt(file_name, delimiter=",")
_,w_test=test_data_raw.shape
test_data=[]
for row in test_data_raw:
    if row[w_test-1] in [0,1]:
        test_data.append(row)
print(len(test_data))
print(len(train_data))


import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import RocCurveDisplay
from numpy import mean, absolute
import warnings
warnings.filterwarnings('ignore')

#######################################
# function for mean deviation
#######################################

# converting tsv file into csv
train_data_arr = np.array(train_data)
test_data_arr = np.array(test_data)

#print(train_data)
#print("***************")
#print(test_data)

# obtain the number of row and column
row_len = train_data_arr.shape[0]
col_len = train_data_arr.shape[1]
X_train = train_data_arr[0:row_len, 0:col_len-1]
y_train = train_data_arr[0:row_len, col_len-1]
row_len1 = test_data_arr.shape[0]
col_len1 = test_data_arr.shape[1]
X_test = test_data_arr[0:row_len1, 0:col_len1-1]
y_test = test_data_arr[0:row_len1, col_len1-1]


######################################
# plot PR curve
######################################

ada = AdaBoostClassifier(n_estimators=10, random_state=0)
ada.fit(X_train, y_train)
neigh = KNeighborsClassifier(n_neighbors=9, weights="uniform")
neigh.fit(X_train, y_train)
rf = RandomForestClassifier(max_features=0.5)
rf.fit(X_train, y_train)
svm = make_pipeline(StandardScaler(), LinearSVC(
    loss="squared_hinge", random_state=0, tol=1e-5))
svm.fit(X_train, y_train)
dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(X_train, y_train)
#plot_precision_recall_curve(ada, X_test, y_test, ax = plt.gca(),name="n={}".format(n_estimators))
display = PrecisionRecallDisplay.from_estimator(
    ada, X_test, y_test, ax=plt.gca())
display = PrecisionRecallDisplay.from_estimator(
    neigh, X_test, y_test, ax=plt.gca())
display = PrecisionRecallDisplay.from_estimator(
    rf, X_test, y_test, ax=plt.gca())
display = PrecisionRecallDisplay.from_estimator(
    svm, X_test, y_test, ax=plt.gca())
display = PrecisionRecallDisplay.from_estimator(
    dummy_clf, X_test, y_test, ax=plt.gca())
display.ax_.set_title(
    "2-class Precision-Recall curve\nThe greater the area under the curve, the better the model")
plt.savefig("./img/models_pr.png")
plt.show()
##############################################
# ROC curve
display = RocCurveDisplay.from_estimator(ada, X_test, y_test, ax=plt.gca())
display = RocCurveDisplay.from_estimator(neigh, X_test, y_test, ax=plt.gca())
display = RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=plt.gca())
display = RocCurveDisplay.from_estimator(svm, X_test, y_test, ax=plt.gca())
display = RocCurveDisplay.from_estimator(
    dummy_clf, X_test, y_test, ax=plt.gca())
display.ax_.set_title(
    "2-class ROC curve\nThe greater the area under the curve, the better the model")
plt.savefig("./img/models_roc.png")
plt.show()

