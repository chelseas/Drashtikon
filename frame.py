from frame_mod import *
from learn import *
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser(description='Framework that trains all learning options.')
parser.add_argument('class1Directory' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('class2Directory' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('--hog' , action="store_true", default=False, help='Input Photo Directory, relative path')
parser.add_argument('--bright' , action="store_true", default=False, help='Input Photo Directory, relative path')
parser.add_argument('--random' , action="store_true", default=False, help='Input Photo Directory, relative path')
parser.add_argument('--all' , action="store_true", help='Runs all classifiers, skipping user input ')
args = parser.parse_args()

def calculateError(predictions, testLabels, msg=''):
  sum_error = 0
  for i in range(predictions.size):
    if predictions[i] != testLabels[i]:
        sum_error += 1
  error = float(sum_error)/float(len(testLabels))
  print(("\t {} Error is " + str(error)).format(msg))
  return error

def main():
    if not args.all:
      selectedModels = selectModels(MODELS)
    else:
      selectedModels = range(len(MODELS))

    # k-fold cross validation
    folds = 5
    num_models = 6
    error_matrix = np.zeros((3,folds,num_models))
    for k in range(1,folds+1):
      path = os.getcwd()
      inputPath1 = os.path.join(path, args.class1Directory, "train_k"+str(k))
      inputPath2 = os.path.join(path, args.class2Directory, "train_k"+str(k))
      testPath1 = os.path.join(path, args.class1Directory, "test_k"+str(k))
      testPath2 = os.path.join(path, args.class2Directory, "test_k"+str(k))

      (rawTrainData1, trainLabels1) = importData(inputPath1, 1)
      (rawTrainData2, trainLabels2) = importData(inputPath2, 2)
      (rawTestData1, testLabels1) = importData(testPath1, 1)
      (rawTestData2, testLabels2) = importData(testPath2, 2)

      printInputStats(inputPath1, inputPath2, testPath1,  testPath2, len(rawTrainData1), len(rawTrainData2), len(rawTestData1), len(rawTestData2))

      if args.hog:
        trainFeatures = getHogFeatures(rawTrainData1 + rawTrainData2, "train data")
        testFeatures1= getHogFeatures(rawTestData1, "test data")
        testFeatures2= getHogFeatures(rawTestData2, "test data")
      elif args.bright:
        trainFeatures = np.array(getMeanBrightness(rawTrainData1 + rawTrainData2, "train data")).reshape(-1, 1)
        testFeatures1= np.array(getMeanBrightness(rawTestData1, "test data")).reshape(-1, 1)
        testFeatures2= np.array(getMeanBrightness(rawTestData2, "test data")).reshape(-1, 1)
      elif args.random:
        trainFeatures = getRandomFeatures(rawTrainData1 + rawTrainData2, "train data")
        testFeatures1= getRandomFeatures(rawTestData1, "test data")
        testFeatures2= getRandomFeatures(rawTestData2, "test data")
      else:
        print("No feature type selected. Exiting...")
        exit(1)

      trainLabels = trainLabels1 + trainLabels2
      testLabels1 = trainLabels1
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
        error_matrix[:,k-1,i] = [testError1, testError2, trainError]

    # print average error over all folds
    # diseases,model,feature,test1error,test2error,training,,
    avg_err = np.sum(error_matrix,axis=1)/float(folds)
    #with open('error.csv','w') as csvfile:
    #    writer = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    #    writer.writerow(['Diseases','Model','Features','Test1 Error','Test2 Error','Train Error'])
    for j in selectedModels:
        with open('error.csv','a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['osd_ptosis',str(MODELS[j].__name__),'only hog rn',str(avg_err[0,j]),str(avg_err[1,j]),str(avg_err[2,j])])

if __name__ == '__main__':
  main()
