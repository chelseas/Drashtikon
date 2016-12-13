# PCA feature dimension selection

from frame_mod import *
from learn import *
import numpy as np
import re
import argparse
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser(description='Framework that trains all learning options.')
parser.add_argument('dataDirectory' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('--hog' , action="store_true", default=True, help='Use hog features. If more than one feature selected, only first used.')
parser.add_argument('--all' , action="store_true", help='Runs all classifiers, skipping user input ')
args = parser.parse_args()

def main():
    # return indices of models being used
    if not args.all:
      selectedModels = [5]
    else:
      selectedModels = range(len(MODELS))

    trainLabels = []
    testLabels = []
    trainData = []
    testData = []
    classNames = []

    path = os.getcwd()
    inputDir = args.dataDirectory #os.path.normpath(os.path.join(path, 
    diseaseClassFolders = [os.path.join(inputDir, name) for name in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, name))]
    for label, diseasePath in enumerate(diseaseClassFolders):
      print(type(label))
      classNames.append(getDiseaseName(diseasePath.split(os.sep)[-1]))
      trainPath = os.path.join(diseasePath, "train")
      testPath = os.path.join(diseasePath, "test")
      (rawTrainData, classTrainLabels) = importData(trainPath, label)
      (rawTestData, classTestLabels) = importData(testPath, label)
      trainLabels = trainLabels + classTrainLabels
      testLabels = testLabels + classTestLabels
      trainData = trainData + rawTrainData
      testData = testData + rawTestData
      printMulticlassInputStats(trainPath, len(classTrainLabels), testPath, len(classTestLabels), label)

    # Custom label for identifying output file easily.
    TARGET = os.path.split(inputDir)[1]
    OUTPUT_ID = datetime.now().strftime("%m%d%H%M%S") + '_' + TARGET
    FEATURE = None

    if args.hog:
      trainFeatures = getHogFeatures(trainData, "train data")
      testFeatures= getHogFeatures(testData, "test data")
      OUTPUT_ID = OUTPUT_ID+"_hog"
      FEATURE = "hog"
    else:
      print("No feature type selected")
      exit(1)

    results = []
    for i in selectedModels:
      print("[ {} ] {}".format(i, MODELS[i].__name__))

      # do a hyper-parameter search and then get the error
      # first plot the PCA breakdown
      pca = decomposition.PCA()
      model = linear_model.LogisticRegression(class_weight='balanced')
      pca.fit(trainFeatures)
      plt.figure(1)
      plt.plot(pca.explained_variance_ratio_)
      plt.xlabel('n_components')
      plt.ylabel('explained variance ratio')
      plt.title('PCA')
      plt.savefig('PCA.png', dpi=300)

      #clf = GridSearchCV(model)


      #model = MODELS[i](trainFeatures, trainLabels)
      #trainPredictions = model.predict(trainFeatures)
      #testPredictions = model.predict(testFeatures)
      #testError = calculateError(testPredictions, testLabels, 'Test')
      #trainError = calculateError(trainPredictions, trainLabels, 'Train')
      #results.append((trainError, testError, MODELS[i].__name__, FEATURE))
      #if args.plot or args.saveplot:
    #    plotConfusionMatrix(testLabels, testPredictions, classNames, args.saveplot, timestamp=OUTPUT_ID, modelName=MODELS[i].__name__)
     # writeOverallResultsToCSV(results, OUTPUT_ID)

if __name__ == '__main__':
  main()




#iris = datasets.load_iris()
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svr = svm.SVC()
#clf = GridSearchCV(svr, parameters)
#clf.fit(iris.data, iris.target)
