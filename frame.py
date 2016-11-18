from frame_mod import *
from SVM import *
import numpy
import argparse

parser = argparse.ArgumentParser(description='Framework that trains all learning options.')
parser.add_argument('class1input' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('class2input' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('testSet' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('testClass' , action="store", help='Input integer for class 1 or class 2')
args = parser.parse_args()

MODELS = [rbfSVM, linearSVM]

def calculateError(predictions, testLabels, msg=''):
  sum_error = 0
  for i in range(len(testLabels)):
    if predictions[i] != testLabels[i]:
        sum_error += 1
  error = float(sum_error)/float(len(testLabels))
  print(("\t {} Error is " + str(error)).format(msg)) 
  return error


def printInputStats(in1, in2, test, train1, train2, testData, testClass):
  print("Class 1 Training Set: {} examples from {}".format(train1, os.path.basename(in1)))
  print("Class 2 Training Set: {} examples from {}".format(train2, os.path.basename(in2)))
  print("Test Set: Class {} with {} examples from {}".format(testClass, train2, os.path.basename(test)))

def main():
  path = os.getcwd()
  inputPath1 = os.path.join(path, args.class1input)
  inputPath2 = os.path.join(path, args.class2input)
  testPath = os.path.join(path, args.testSet)
  testClass= int(args.testClass)
  #(train_data,train_labels,test_data,test_labels) = import_data(inputPath1, inputPath2)
  (rawTrainData1, trainLabels1) = importData(inputPath1, 1)
  (rawTrainData2, trainLabels2) = importData(inputPath2, 2)
  (rawTestData, testLabels) = importData(testPath, testClass)

  printInputStats(inputPath1, inputPath2, testPath, len(rawTrainData1), len(rawTrainData2), len(rawTestData), args.testClass)

  trainFeatures = getHogFeatures(rawTrainData1 + rawTrainData2, "train data")
  trainLabels = trainLabels1 + trainLabels2
  testFeatures = getHogFeatures(rawTestData, "test data")

  for m in MODELS:
    model = m(trainFeatures, trainLabels)
    testPredictions = model.predict(testFeatures)
    trainPredictions = model.predict(trainFeatures)
    testError = calculateError(testPredictions, testLabels, 'Test')
    trainError = calculateError(trainPredictions, trainLabels, 'Train')
    


if __name__ == '__main__':
  main()



