import os
import argparse
from shutil import copy
from random import shuffle
from datetime import datetime

parser = argparse.ArgumentParser(description='Divides cropped data into test and train data.')
parser.add_argument('inputFolder' , action="store", help='Input Photo Directory, relative path to script directory')
parser.add_argument('testPercent', action="store", help='Decimal representing fraction to put in test set')
args = parser.parse_args()


def main():
  path = os.getcwd()
  target = os.path.join(path, args.inputFolder)
  imgFiles = [f for f in os.listdir(target) if (os.path.isfile(os.path.join(target, f)) and any(f.endswith(ext) for ext in ['.jpg', '.JPG', '.png']))]
  ntest = int(len(imgFiles) *float(args.testPercent))
  shuffle(imgFiles)
  ouptutDir = os.path.join(target,'..', os.path.basename(target) + "_partition_"+datetime.now().strftime("%m%d%H%M%S"))
  ouptutDir_test = os.path.join(ouptutDir, "test")
  ouptutDir_train = os.path.join(ouptutDir, "train")
  os.mkdir(ouptutDir)
  os.mkdir(ouptutDir_test)
  os.mkdir(ouptutDir_train)
  for i in range(ntest):
    copy(os.path.join(target, imgFiles[i]), ouptutDir_test)
    # print(imgFiles[i])
    # print(ouptutDir_test)
  for i in range(ntest, len(imgFiles)):
    copy(os.path.join(target, imgFiles[i]), ouptutDir_train)
    # print(imgFiles[i])
    # print(ouptutDir_train)



if __name__ == '__main__':
  main()
