from frame_mod import *
from learn import *
import numpy as np
import re
import argparse
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

parser = argparse.ArgumentParser(description='Framework that trains all learning options.')
parser.add_argument('dataDirectory' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('--hog' , action="store_true", default=False, help='Use hog features. If more than one feature selected, only first used.')
parser.add_argument('--pcahog' , action="store_true", default=False, help='Use PCA hog features. If more than one feature selected, only first used.')
parser.add_argument('--croppedhog' , action="store_true", default=False, help='Used cropped hog features.')
parser.add_argument('--pcacroppedhog' , action="store_true", default=False, help='Used cropped hog features.')
parser.add_argument('--daisy' , action="store_true", default=False, help='Use DAISY features, similar to sift.')
parser.add_argument('--bright' , action="store_true", default=False, help='Use mean brightness feature. If more than one feature selected, only first used.')
parser.add_argument('--random' , action="store_true", default=False, help='Use random subset of hog features. If more than one feature selected, only first used.')
parser.add_argument('--all' , action="store_true", help='Runs all classifiers, skipping user input ')
parser.add_argument('--plot' , action="store_true", help='Draws a confusion matrix')
parser.add_argument('--saveplot' , action="store_true", help='Saves a confusion matrix')
args = parser.parse_args()

def main():
    # return indices of models being used
    if not args.all:
      selectedModels = selectModels(MODELS)
    else:
      selectedModels = range(len(MODELS))

    trainLabels = []
    testLabels = []
    trainData = []
    testData = []
    classNames = []

    path = os.getcwd()
    inputDir = os.path.normpath(os.path.join(path, args.dataDirectory))
    diseaseClassFolders = [os.path.join(inputDir, name) for name in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, name))]
    for label, diseasePath in enumerate(diseaseClassFolders):
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
    elif args.pcahog:
      trainFeatures, testFeatures = getPCAHogFeatures(trainData=trainData, testData=testData, msg="train/test data", nComponents=len(trainData))
      OUTPUT_ID = OUTPUT_ID+"_pcahog"
      FEATURE = "pcahog"
    elif args.bright:
      trainFeatures = np.array(getMeanBrightness(trainData, "train data")).reshape(-1, 1)
      testFeatures= np.array(getMeanBrightness(testData, "test data")).reshape(-1, 1)
      OUTPUT_ID = OUTPUT_ID+"_bright"
      FEATURE = "bright"
    elif args.random:
      trainFeatures = getRandomFeatures(trainData, "train data")
      testFeatures= getRandomFeatures(testData, "test data")
      OUTPUT_ID = OUTPUT_ID+"_random"
      FEATURE = "random"
    elif args.croppedhog:
      trainFeatures = getCroppedHogFeatures(trainData, "train data")
      testFeatures= getCroppedHogFeatures(testData, "test data")
      OUTPUT_ID = OUTPUT_ID+"_croppedhog"
      FEATURE = "croppedhog"
    elif args.daisy:
      trainFeatures = getDaisyFeatures(trainData, "train data")
      testFeatures= getDaisyFeatures(testData, "test data")
      OUTPUT_ID = OUTPUT_ID+"_daisy"
      FEATURE = "daisy"
    elif args.pcacroppedhog:
      trainFeatures, testFeatures = getPCACroppedHogFeatures(trainData=trainData, testData=testData, msg="train/test data", nComponents=len(trainData))
      OUTPUT_ID = OUTPUT_ID+"_pcacroppedhog"
      FEATURE = "pcacroppedhog"
    else:
      print("No feature type selected")
      exit(1)

    results = []
    for i in selectedModels:
      print("[ {} ] {}".format(i, MODELS[i].__name__))
      model = MODELS[i](trainFeatures, trainLabels)
      #y_score_train = model.decision_function(trainFeatures)
      ylabel_binary = label_binarize(testLabels, classes=[0, 1, 2, 3, 4])
      y_score_test = model.decision_function(testFeatures)
      print(y_score_test)
      fpr = dict()
      tpr = dict()
      roc_auc = dict()
      #plt.figure()
      for i in range(5):
          fpr[i], tpr[i], _ = roc_curve(ylabel_binary[:,i], y_score_test[:,i])
          roc_auc[i] = auc(fpr[i], tpr[i])
          plt.plot(fpr[i],tpr[i], label='ROC curve with area %0.2f for class %s' % (roc_auc[i], classNames[i]) )
      fig_str = 'ROC ' + str(MODELS[i].__name__) + ', ' + FEATURE
      plt.savefig(fig_str+'.png', dpi=300)


      trainPredictions = model.predict(trainFeatures)
      testPredictions = model.predict(testFeatures)
      testError = calculateError(testPredictions, testLabels, 'Test')
      trainError = calculateError(trainPredictions, trainLabels, 'Train')
      results.append((trainError, testError, MODELS[i].__name__, FEATURE))
      if args.plot or args.saveplot:
        plotConfusionMatrix(testLabels, testPredictions, classNames, args.saveplot, timestamp=OUTPUT_ID, modelName=MODELS[i].__name__)
      writeOverallResultsToCSV(results, OUTPUT_ID)

if __name__ == '__main__':
  main()
