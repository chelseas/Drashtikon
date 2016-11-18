# module for frame.py
import os, sys
import re
from PIL import Image
from random import *
from math import *
import numpy as np
import copy
from skimage.feature import hog

p = re.compile('.*[.]jpg',re.IGNORECASE)

def getHogFeatures(data, msg="data"):
    print("Extracting HOG features for "+ msg + "...")
    result = np.array([hog(x) for x in data])
    print("Done.")
    return result


def importData(path, label):
    data = []
    for filename in os.listdir(path):
        if p.match(filename):
            img = np.array(Image.open(os.path.join(path, filename)))
            data.append(img)
    labels = [label for x in range(len(data))]
    return (data, labels)



def import_data(folder1, folder2):
    # for each file in the folders, import the images and add to array
    loc1 = folder1
    list1 = []
    for filename in os.listdir(loc1):
        if p.match(filename):
            img = np.array(Image.open(os.path.join(loc1, filename)))
            list1.append(img)
    labels1 = [1 for x in range(len(list1))]

    loc2 = folder2
    list2 = []
    for filename in os.listdir(loc2):
        if p.match(filename):
            img = np.array(Image.open(os.path.join(loc2, filename)))
            list2.append(img)
    labels2 = [2 for x in range(len(list2))]

    data = list1+list2
    labels = labels1+labels2
    # test_data = copy.deepcopy(train_data)
    # test_labels = copy.deepcopy(train_labels)
    return (data,labels)
