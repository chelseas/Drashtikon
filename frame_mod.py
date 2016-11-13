# module for frame.py
from pathlib import Path
import os, sys
import re
from PIL import Image
from random import *
from math import *

seed(0)
p = re.compile('.*[.]jpg',re.IGNORECASE)

def import_data(folder1, folder2):
    # ##################
    # for each file in the folders, import the images and add to array
    loc1 = os.getcwd() + '/' + folder1
    list1 = []
    for filename in os.listdir(loc1):
        if p.match(filename):
            list1.append(Image.open(loc1+'/'+filename))

    labels1 = [1 for x in range(len(list1))]
    # ##################
    loc2 = os.getcwd() + '/' + folder2
    list2 = []
    for filename in os.listdir(loc2):
        if p.match(filename):
            list2.append(Image.open(loc2+'/'+filename))

    labels2 = [2 for x in range(len(list2))]
    # ##################
    train_data = list1+list2
    train_labels = labels1+labels2
    # ##################
    # randomly select examples to go into train and test
    test_data = []
    test_labels = []
    # can change. right now selecting 10% for testing
    for i in range(ceil(len(train_labels)/10)):
        ind = randint(0,len(train_labels)-1)
        test_labels.append(train_labels[ind])
        test_data.append(train_data[ind])
        del train_data[ind]
        del train_labels[ind]
    # ##################
    return (train_data,train_labels,test_data,test_labels)
