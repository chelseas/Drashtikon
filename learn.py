# Learning Method: SVM
from scipy import *
from sklearn import svm
from sklearn import linear_model



# RBF SVMs
def rbfSVM():
    model = svm.SVC(kernel='rbf')
    return model

def rbfSVMBalanced():
    model = svm.SVC(kernel='rbf', class_weight='balanced')
    return model


# Linear SVMs
def linearSVM():
    model = svm.SVC(kernel='linear')
    return model

def linearSVMBalanced():
    model = svm.SVC(kernel='linear', class_weight='balanced')
    return model


# Logistic Regression
def logisticRegression():
    model = linear_model.LogisticRegression()
    return model

def logisticRegressionBalanced():
    model = linear_model.LogisticRegression(class_weight='balanced')
    return model

MODELS = [rbfSVM, rbfSVMBalanced, linearSVM,  linearSVMBalanced, logisticRegression, logisticRegressionBalanced]
