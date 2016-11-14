# Learning Method: SVM
from scipy import *
from sklearn import svm

def getHogFeatures(image_array):
    #a = [list(x.getdata(band=0))[0:9] for x in image_array]
    return a

def SVM(train_data, train_labels):
    model = svm.SVC(kernel='rbf')
    model.fit(train_data, train_labels)
    return model.predict
