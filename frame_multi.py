from frame_mod import *
from learn import *
import numpy as np
import re
import argparse
import csv
from datetime import datetime

parser = argparse.ArgumentParser(description='Framework that trains all learning options.')
parser.add_argument('dataDirectory' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('--hog' , action="store_true", default=False, help='Use hog features. If more than one feature selected, only first used.')
parser.add_argument('--croppedhog' , action="store_true", default=False, help='Used cropped hog features.')
parser.add_argument('--daisy' , action="store_true", default=False, help='Use DAISY features, similar to sift.')
parser.add_argument('--bright' , action="store_true", default=False, help='Use mean brightness feature. If more than one feature selected, only first used.')
parser.add_argument('--random' , action="store_true", default=False, help='Use random subset of hog features. If more than one feature selected, only first used.')
parser.add_argument('--all' , action="store_true", help='Runs all classifiers, skipping user input ')
args = parser.parse_args()


def main():
    # return indices of models being used
    if not args.all:
      selectedModels = selectModels(MODELS)
    else:
      selectedModels = range(len(MODELS))

    outFolder = datetime.now().strftime("%m%d%H%M%S")
    os.mkdir(os.path.join(os.getcwd(), 'output', outFolder))
  
    trainLabels = []
    testLabels = []
    trainData = []
    testData = []

    path = os.getcwd()
    inputDir = os.path.join(path, args.dataDirectory)
    diseaseClassFolders = [os.path.join(inputDir, name) for name in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, name))]
    for label, diseasePath in enumerate(diseaseClassFolders):
      trainPath = os.path.join(diseasePath, "train")
      testPath = os.path.join(diseasePath, "test")
      (rawTrainData, classTrainLabels) = importData(trainPath, label)  
      (rawTestData, classTestLabels) = importData(testPath, label)
      trainLabels = trainLabels + classTrainLabels
      testLabels = testLabels + classTestLabels
      trainData = trainData + rawTrainData
      testData = testData + rawTestData
      printMulticlassInputStats(trainPath, len(classTrainLabels), testPath, len(classTestLabels), label)

    if args.hog:
      trainFeatures = getHogFeatures(trainData, "train data")
      testFeatures= getHogFeatures(testData, "test data")
    elif args.bright:
      trainFeatures = np.array(getMeanBrightness(trainData, "train data")).reshape(-1, 1)
      testFeatures= np.array(getMeanBrightness(testData, "test data")).reshape(-1, 1)
    elif args.random:
      trainFeatures = getRandomFeatures(trainData, "train data")
      testFeatures= getRandomFeatures(testData, "test data")
    elif args.croppedhog:
      trainFeatures = getCroppedHogFeatures(trainData, "train data")
      testFeatures= getCroppedHogFeatures(testData, "test data")
    elif args.daisy:
      trainFeatures = getDaisyFeatures(trainData, "train data")
      testFeatures= getDaisyFeatures(testData, "test data")
    else: 
      print("No feature type selected")
      exit(1)

    for i in selectedModels:
      print("[ {} ] {}".format(i, MODELS[i].__name__))
      model = MODELS[i](trainFeatures, trainLabels)
      trainPredictions = model.predict(trainFeatures)
      testPredictions = model.predict(testFeatures)
      testError = calculateError(testPredictions, testLabels, 'Test')
      trainError = calculateError(trainPredictions, trainLabels, 'Train')
      plotConfusionMatrix(testLabels, testPredictions, "")

if __name__ == '__main__':
  main()
