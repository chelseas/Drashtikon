# Learning Method: SVM
from scipy import *
from sklearn import svm
from sklearn import linear_model



# RBF SVMs
def rbfSVM(train_data, train_labels):
    model = svm.SVC(kernel='rbf')
    model.fit(train_data, train_labels)
    return model

def rbfSVMBalanced(train_data, train_labels):
    model = svm.SVC(kernel='rbf', class_weight='balanced')
    model.fit(train_data, train_labels)
    return model


# Linear SVMs
def linearSVM(train_data, train_labels):
    model = svm.SVC(kernel='linear')
    model.fit(train_data, train_labels)
    return model

def linearSVMBalanced(train_data, train_labels):
    model = svm.SVC(kernel='linear', class_weight='balanced')
    model.fit(train_data, train_labels)
    return model


# Logistic Regression
def logisticRegression(train_data, train_labels):
    model = linear_model.LogisticRegression()
    model.fit(train_data, train_labels)
    return model

def logisticRegressionBalanced(train_data, train_labels):
    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(train_data, train_labels)
    return model

MODELS = [rbfSVM, rbfSVMBalanced, linearSVM,  linearSVMBalanced, logisticRegression, logisticRegressionBalanced]
