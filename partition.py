import os
import argparse
from shutil import copy
from random import shuffle
from datetime import datetime

parser = argparse.ArgumentParser(description='Divides cropped data into test and train data.')
parser.add_argument('inputFolder' , action="store", help='Input Photo Directory, relative path to script directory')
#parser.add_argument('testPercent', action="store", help='Decimal representing fraction to put in test set')
args = parser.parse_args()


def main():
  path = os.getcwd()
  print(str(path))
  target = os.path.join(path, args.inputFolder)
  print(str(target))
  imgFiles = [f for f in os.listdir(target) if (os.path.isfile(os.path.join(target, f)) and any(f.endswith(ext) for ext in ['.jpg', '.JPG', '.png']))]
  shuffle(imgFiles) # this line randomizes
  ouptutDir = os.path.join(target,'..', os.path.basename(target) + "_k_fold_partition_"+datetime.now().strftime("%m%d%H%M%S"))
  os.mkdir(ouptutDir)
  fold = 5
  size_partition = int(len(imgFiles)*(1.0/fold))
  max_index = len(imgFiles)
  # store range objects
  ########################
  for k in range(1,fold+1):
      test_subset = range((k-1)*size_partition,(k-1)*size_partition + size_partition)
      train_subset = [i for j in (range(0,(k-1)*size_partition), range((k-1)*size_partition + size_partition, max_index) ) for i in j]
      #ntest = int(size_partition *float(args.testPercent))
      ouptutDir_test = os.path.join(ouptutDir, "test_k"+str(k))
      ouptutDir_train = os.path.join(ouptutDir, "train_k"+str(k))
      os.mkdir(ouptutDir_test)
      os.mkdir(ouptutDir_train)
      for i in test_subset:
        copy(os.path.join(target, imgFiles[i]), ouptutDir_test)
        # print(imgFiles[i])
        # print(ouptutDir_test)
      for i in train_subset:
        copy(os.path.join(target, imgFiles[i]), ouptutDir_train)
        # print(imgFiles[i])
        # print(ouptutDir_train)



if __name__ == '__main__':
  main()
