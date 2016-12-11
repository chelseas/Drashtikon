# module for frame.py
import os, sys
import re
from PIL import Image, ImageStat
from random import *
from math import *
import numpy as np
import copy
from skimage.feature import hog, daisy
import sys
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

p = re.compile('.*[.]jpg',re.IGNORECASE)

def getHogFeatures(data, msg="data"):
    print("Extracting HOG features for "+ msg + "...")
    result = np.array([hog(x) for x in data])
    print("Done.")
    return result

def getMeanBrightness(data, msg="data"):
    print("Extracting Mean Brightness for "+ msg +"...")
    result = [ImageStat.Stat(Image.fromarray(x)).mean[0] for x in data]
    print("Done.")
    return result

def getRandomFeatures(data, msg="data"):
    print("Extracting Random hog features for "+ msg+ "...")
    result = np.array([hog(x[len(x)/4: len(x)/2]) for x in data])
    print("Done")
    return result

def getCroppedHogFeatures(data, msg="data"):
  print("Extracting Cropped HOG features for "+ msg +"...")
  result = []
  for x in data:
    img = Image.fromarray(x)
    width = img.size[0]
    newDim = 90
    offsetX = 35
    offsetY = 5
    img_l = img.crop((offsetX, offsetY, offsetX+newDim, newDim + offsetY))
    img_r = img.crop((width-newDim-offsetX, offsetY, width-offsetX, newDim + offsetY))
    assert(img_l.size[0] == img_r.size[0])
    assert(img_l.size[1] == img_r.size[1])
    featuresL = hog(np.array(img_l))
    featuresR = hog(np.array(img_r))
    combinedFeatures = np.concatenate((featuresL, featuresR))
    result.append(combinedFeatures)
  print("Done")
  return result

def getDaisyFeatures(data, msg="data"):
    print("Extracting DAISY features for "+ msg+ "...")
    result = np.array([daisy(x) for x in data])
    print("Done")
    return result

def importData(path, label):
    data = []
    for filename in os.listdir(path):
        if p.match(filename):
            img = np.array(Image.open(os.path.join(path, filename)))
            data.append(img)
    labels = [label for x in range(len(data))]
    return (data, labels)


def printInputStats(train1, train2, test1, test2, ntrain1, ntrain2, ntest1, ntest2):
  print("Training Set: ")
  print("\t Class 1 -- {} examples from {}".format(ntrain1, train1))
  print("\t Class 2 -- {} examples from {}".format(ntrain2, train2))
  print("Test Set: ")
  print("\t Class 1 -- {} examples from {}".format(ntest1, test1))
  print("\t Class 2 -- {} examples from {}".format(ntest2, test2))

def printMulticlassInputStats(trainDir, nTrain, testDir, nTest, label):
  print("Class {} -----").format(label)
  print("\t Training: {} examples from {}".format(nTrain, trainDir))
  print("\t Test: {} examples from {}".format(nTest, testDir))


def selectModels(MODELS):
    print("\n Type numbered indexes separated by spaces to select models to train.")
    print("Enter an empty line to train all \n")
    for i, m in enumerate(MODELS):
        print("[ {} ] -- {}".format(i, m.__name__))
    resp = [int(c.rstrip()) for c in sys.stdin.readline().split(' ') if c.rstrip().isdigit() and 0 <= int(c.rstrip()) < len(MODELS)]
    if len(resp) == 0:
        resp = range(len(MODELS))
    print("Selected the following models {} \n".format(resp))
    return resp

def calculateError(predictions, testLabels, msg=''):
  sum_error = 0
  for i in range(predictions.size):
    if predictions[i] != testLabels[i]:
        sum_error += 1
  error = float(sum_error)/float(len(testLabels))
  print(("\t {} Error is " + str(error)).format(msg))
  return error


def plotConfusionMatrix(trueLabels, predLabels, classNames):
      cnf_matrix = confusion_matrix(trueLabels, predLabels)
      np.set_printoptions(precision=2)
      plt.figure()
      drawConfusionMatrix(cnf_matrix, classes=["class1", "class2"], title='Confusion matrix, without normalization')
      plt.figure()
      drawConfusionMatrix(cnf_matrix, classes=["class1", "class2"], normalize=True, title='Normalized confusion matrix')
      plt.show()



def drawConfusionMatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')