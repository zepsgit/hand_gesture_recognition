import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def train_model_arc(model):
    with open("./dataset/catenate_dataset.csv") as file_name:
        cat_data = np.loadtxt(file_name, delimiter=",")
    with open("./dataset/arclen_dataset.csv") as file_name:
        arc_data = np.loadtxt(file_name, delimiter=",")
    h,w=cat_data.shape
    X_arc=arc_data[:,0:w-1]
    y=cat_data[:,w-1]
    model.fit(X_arc, y)
    return model

def train_model_landmark(model):
    with open("./dataset/dataset12.csv") as file_name:
        data = np.loadtxt(file_name, delimiter=",")
    h,w=data.shape
    X=data[:,0:w-1]
    y=data[:,w-1]
    print(X[0])
    model.fit(X, y)
    return model

arc_rf=train_model_arc(RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0))
arc_knn=train_model_arc(KNeighborsClassifier(n_neighbors=3))
lm_rf=train_model_landmark(RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0))
lm_knn=train_model_landmark(KNeighborsClassifier(n_neighbors=3))
