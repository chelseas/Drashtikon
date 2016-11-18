# Learning Method: SVM
from scipy import *
from sklearn import svm

def rbfSVM(train_data, train_labels):
    print("Model: RBF SVM")
    model = svm.SVC(kernel='rbf')
    model.fit(train_data, train_labels)
    return model

def linearSVM(train_data, train_labels):
    print("Model: Linear SVM")
    model = svm.SVC(kernel='linear')
    model.fit(train_data, train_labels)
    return model

