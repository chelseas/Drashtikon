from frame_mod import *
from learn import *
import numpy
import argparse

parser = argparse.ArgumentParser(description='Framework that trains all learning options.')
parser.add_argument('trainClass1' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('trainClass2' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('testClass1' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('testClass2' , action="store", help='Input Photo Directory, relative path')
args = parser.parse_args()

def calculateError(predictions, testLabels, msg=''):
  sum_error = 0
  for i in range(len(testLabels)):
    if predictions[i] != testLabels[i]:
        sum_error += 1
  error = float(sum_error)/float(len(testLabels))
  print(("\t {} Error is " + str(error)).format(msg)) 
  return error


def main():
  path = os.getcwd()
  inputPath1 = os.path.join(path, args.trainClass1)
  inputPath2 = os.path.join(path, args.trainClass2)
  testPath1 = os.path.join(path, args.testClass1)
  testPath2 = os.path.join(path, args.testClass2)

  (rawTrainData1, trainLabels1) = importData(inputPath1, 1)
  (rawTrainData2, trainLabels2) = importData(inputPath2, 2)
  (rawTestData1, testLabels1) = importData(testPath1, 1)
  (rawTestData2, testLabels2) = importData(testPath2, 2)

  printInputStats(args.trainClass1, args.trainClass2, args.testClass1, args.testClass2, len(rawTrainData1), len(rawTrainData2), len(rawTestData1), len(rawTestData2))

  selectedModels = selectModels(MODELS)

  trainFeatures = getHogFeatures(rawTrainData1 + rawTrainData2, "train data")
  trainLabels = trainLabels1 + trainLabels2
  
  testFeatures1= getHogFeatures(rawTestData1, "test data")
  testLabels1 = trainLabels1

  testFeatures2= getHogFeatures(rawTestData2, "test data")
  testLabels2 = trainLabels2

  for i in selectedModels:
    print("[ {} ] {}".format(i, MODELS[i].__name__))
    model = MODELS[i](trainFeatures, trainLabels)
    testPredictions1 = model.predict(testFeatures1)
    testPredictions2 = model.predict(testFeatures2)
    trainPredictions = model.predict(trainFeatures)
    testError1 = calculateError(testPredictions1, testLabels1, 'Test Class 1')
    testError2 = calculateError(testPredictions2, testLabels2, 'Test Class 2')
    trainError = calculateError(trainPredictions, trainLabels, 'Train')
    


if __name__ == '__main__':
  main()



