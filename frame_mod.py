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
