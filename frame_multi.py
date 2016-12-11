from frame_mod import *
from learn import *
import numpy as np
import re
import argparse
import csv
from datetime import datetime

parser = argparse.ArgumentParser(description='Framework that trains all learning options.')
parser.add_argument('dataDirectory' , action="store", help='Input Photo Directory, relative path')
#parser.add_argument('class2Directory' , action="store", help='Input Photo Directory, relative path')
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
  
    # the three layers of the error matrix are test error on class 1, test err cls.2 and train error
    # the y-dimension (row) is fold index and the x-dimension (col) is the model index
    #error_matrix = np.zeros((3,1,len(selectedModels)))

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

    # inputPath1 = os.path.join(path, args.class1Directory, "train")
    # inputPath2 = os.path.join(path, args.class2Directory, "train")
    # testPath1 = os.path.join(path, args.class1Directory, "test")
    # testPath2 = os.path.join(path, args.class2Directory, "test")

    # (rawTrainData1, trainLabels1) = importData(inputPath1, 1)
    # (rawTrainData2, trainLabels2) = importData(inputPath2, 2)
    # (rawTestData1, testLabels1) = importData(testPath1, 1)
    # (rawTestData2, testLabels2) = importData(testPath2, 2)

    # trainLabels = trainLabels1 + trainLabels2

    # printInputStats(inputPath1, inputPath2, testPath1,  testPath2, len(rawTrainData1), len(rawTrainData2), len(rawTestData1), len(rawTestData2))

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


    # print average error over all folds
    # ['Diseases','Model','Features','Eval Set','Error Value']
    #avg_err = np.sum(error_matrix,axis=1)
    #print(avg_err)
    # with open(file_to_write_to,'w') as csvfile:
    #   writer = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    #   for j in selectedModels:
    #     writer.writerow([dis_set,str(MODELS[j].__name__),feat,diseases[0],str(avg_err[0,j])])
    #     writer.writerow([dis_set,str(MODELS[j].__name__),feat,diseases[1],str(avg_err[1,j])])
    #     writer.writerow([dis_set,str(MODELS[j].__name__),feat,'train',str(avg_err[2,j])])

if __name__ == '__main__':
  main()
