# module for frame.py
import os, sys
import re
from PIL import Image
from random import *
from math import *
import numpy as np
import copy

seed(0)
p = re.compile('.*[.]jpg',re.IGNORECASE)

def import_data(folder1, folder2):
    # ##################
    # for each file in the folders, import the images and add to array
    loc1 = folder1
    list1 = []
    for filename in os.listdir(loc1):
        if p.match(filename):
            img = np.array(Image.open(os.path.join(loc1, filename)))
            list1.append(img)
    labels1 = [1 for x in range(len(list1))]
    # ##################
    loc2 = folder2
    list2 = []
    for filename in os.listdir(loc2):
        if p.match(filename):
            img = np.array(Image.open(os.path.join(loc2, filename)))
            list2.append(img)
    labels2 = [2 for x in range(len(list2))]
    # ##################
    train_data = list1+list2
    train_labels = labels1+labels2
    # ##################
    # TODO: randomly select examples to go into train and test
    # test_data = []
    # test_labels = []
    # # can change. right now selecting 10% for testing
    # for i in range(ceil(len(train_labels)/10)):
    #     ind = randint(0,len(train_labels)-1)
    #     test_labels.append(train_labels[ind])
    #     test_data.append(train_data[ind])
    #     del train_data[ind]
    #     del train_labels[ind]
    # ##################
    test_data = copy.deepcopy(train_data)
    test_labels = copy.deepcopy(train_labels)
    return (train_data,train_labels,test_data,test_labels)
