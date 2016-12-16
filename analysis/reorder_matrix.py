
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import os
import csv
import sys
sys.path.append('/Users/zmaurer/Documents/Academics/2016_2017/CS229/final_project/code/ML-party/')
import frame_mod

# path1 = os.path.abspath('f1data/hog')
# path2 = os.path.abspath('f1data/croppedhog')
# path3 = os.path.abspath('f1data/pcacroppedhog')

disease_names = ['dermoid_cyst', 'goonderson_flap', 'osd', 'ptosis', 'str']
re_mapping = [2, 4, 0, 3, 1]

csv_in = 'finalCV/confusion_matrix_rbfCVSVM.csv'

# 
csvfile = os.path.abspath(csv_in)
my_data = np.genfromtxt(csvfile, delimiter=',')
true = []
pred = []
for i in range(5):
    for j in range(5):
        n = int(my_data[i,j])
        true = true + [re_mapping[i]]*n
        pred = pred + [re_mapping[j]]*n
frame_mod.plotConfusionMatrix(true, pred, disease_names, True, "_reorder_final", "rbfCVSVM")