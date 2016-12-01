from frame_mod import *
from learn import *
import numpy as np
import re
import argparse
import csv
import sklearn
from matplotlib.pyplot import *

parser = argparse.ArgumentParser(description='Framework that trains all learning options.')
parser.add_argument('class1Directory' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('class2Directory' , action="store", help='Input Photo Directory, relative path')
parser.add_argument('number_k_folds', action="store", default=1, help='Input number of k-folds if you''d like to use for k-fold cross validation')
parser.add_argument('--hog' , action="store_true", default=False, help='Use hog features. If more than one feature selected, only first used.')
parser.add_argument('--bright' , action="store_true", default=False, help='Use mean brightness feature. If more than one feature selected, only first used.')
parser.add_argument('--random' , action="store_true", default=False, help='Use random subset of hog features. If more than one feature selected, only first used.')
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

# get disease name from file path using regexps
str_p = re.compile('(str)',re.IGNORECASE)
ptosis_p = re.compile('(ptosis)',re.IGNORECASE)
osd_p = re.compile('(osd)',re.IGNORECASE)
#pd4 = re.compile('(NEXT_NEW_DISEASE)',re.IGNORECASE)
dirs = [args.class1Directory, args.class2Directory]
diseases = ['hi','mom']
for i in range(2):
    dir2srch = dirs[i]
    if str_p.search(dir2srch) and (not osd_p.search(dir2srch)) and (not ptosis_p.search(dir2srch)):
        diseases[i] = 'str'
    elif (not str_p.search(dir2srch)) and osd_p.search(dir2srch) and (not ptosis_p.search(dir2srch)):
        diseases[i] = 'osd'
    elif (not str_p.search(dir2srch)) and (not osd_p.search(dir2srch)) and ptosis_p.search(dir2srch):
        diseases[i] = 'ptosis'
    else:
        print("Broski, your file paths are confusing. IDK what disease you're tryna classify. Make sure only 1 disease name is in the relative path.")
        exit(1)

if diseases[0]=='hi' or diseases[1]=='mom':
    print('Broski, you messed up. I couldn''t identify both the diseases from your file paths')
    exit(1)
else:
    dis_set = diseases[0] + '_' + diseases[1]

def main():
    # return indices of models being used
    if not args.all:
      selectedModels = selectModels(MODELS)
    else:
      selectedModels = range(len(MODELS))

    if args.hog:
        feat="hog"
        file_to_write_to = 'error_hog.csv'
    elif args.bright:
        feat="bright"
        file_to_write_to = 'error_bright.csv'
    elif args.random:
        feat="random"
        file_to_write_to = 'error_random.csv'
    else:
        print("No feature type selected. Exiting...")
        exit(1)

    # k-fold cross validation
    folds = int(args.number_k_folds)
    # this is the total number of models possible. may be more than number
    # currently being run
    num_models = len(MODELS)

    path = os.getcwd()
    inputPath1 = os.path.join(path, args.class1Directory)
    inputPath2 = os.path.join(path, args.class2Directory)
#    testPath1 = os.path.join(path, args.class1Directory)
#    testPath2 = os.path.join(path, args.class2Directory)

    (rawTrainData1, trainLabels1) = importData(inputPath1, 1)
    (rawTrainData2, trainLabels2) = importData(inputPath2, 2)
#    (rawTestData1, testLabels1) = importData(testPath1, 1)
#    (rawTestData2, testLabels2) = importData(testPath2, 2)

    trainLabels = trainLabels1 + trainLabels2
    #print(trainLabels1[0:5],' , ',trainLabels2[0:5])
    #mid = len(trainLabels1)
    #print(trainLabels[0:5], trainLabels[mid:mid+5])
    #printasdfasdfa
    trainLabels = np.array(trainLabels)
    #print(trainLabels.shape)

    #printInputStats(inputPath1, inputPath2, testPath1,  testPath2, len(rawTrainData1), len(rawTrainData2), len(rawTestData1), len(rawTestData2))

    if feat=="hog":
        trainFeatures = getHogFeatures(rawTrainData1 + rawTrainData2, "train data")
#        testFeatures1= getHogFeatures(rawTestData1, "test data")
#        testFeatures2= getHogFeatures(rawTestData2, "test data")
    elif feat=="bright":
        trainFeatures = np.array(getMeanBrightness(rawTrainData1 + rawTrainData2, "train data")).reshape(-1, 1)
#        testFeatures1= np.array(getMeanBrightness(rawTestData1, "test data")).reshape(-1, 1)
#        testFeatures2= np.array(getMeanBrightness(rawTestData2, "test data")).reshape(-1, 1)
    elif feat=="random":
        trainFeatures = getRandomFeatures(rawTrainData1 + rawTrainData2, "train data")
#        testFeatures1= getRandomFeatures(rawTestData1, "test data")
#        testFeatures2= getRandomFeatures(rawTestData2, "test data")

#      testLabels1 = trainLabels1 # Better question: why didn't this throw an error? train and test labels were different sizes
#      testLabels2 = trainLabels2
    #print(trainFeatures.shape)
    #print(trainLabels.shape)

    for i in selectedModels:
        print("[ {} ] {}".format(i, MODELS[i].__name__))
        #model = MODELS[i](trainFeatures, trainLabels)
        #testPredictions1 = model.predict(testFeatures1)
        #testPredictions2 = model.predict(testFeatures2)
        #trainPredictions = model.predict(trainFeatures)
        #testError1 = calculateError(testPredictions1, testLabels1, 'Test Class 1')
        #testError2 = calculateError(testPredictions2, testLabels2, 'Test Class 2')
        #trainError = calculateError(trainPredictions, trainLabels, 'Train')
        #error_matrix[:,k-1,i] = [testError1, testError2, trainError]
        model = MODELS[i]()
        train_fracs = np.linspace(.75,1.0,7)
        #print(train_fracs)
        # default is 3 folds
        train_sz, train_sc, cv_score = sklearn.model_selection.learning_curve(model, trainFeatures, trainLabels, train_sizes = train_fracs, n_jobs=1, cv=folds)
        train_err = train_sc.sum(axis=1)/float(folds)
        cv_err = cv_score.sum(axis=1)/float(folds)
        fig = figure(i)
        l1, = plot(train_sz, train_err,'k--', label='Training Error' )
        hold(True)
        l2, = plot(train_sz, cv_err, 'g^', label='CV Error (Test)')
        hold(False)
        legend(handles=[l1, l2])
        xlabel('Number of training examples')
        ylabel('Accuracy')
        t_str = 'Error for ' + str(MODELS[i].__name__) + ' ' + dis_set + ' ' + feat
        title(t_str)
        fig.savefig(t_str+'.png')

    # show plots
    #show()

    # print average error over all folds
    # ['Diseases','Model','Features','Eval Set','Error Value']
    #avg_err = np.sum(error_matrix,axis=1)/float(folds)
    #for j in selectedModels:
    #    with open(file_to_write_to,'a') as csvfile:
    #        writer = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    #        writer.writerow([dis_set,str(MODELS[j].__name__),feat,diseases[0],str(avg_err[0,j])])
    #        writer.writerow([dis_set,str(MODELS[j].__name__),feat,diseases[1],str(avg_err[1,j])])
    #        writer.writerow([dis_set,str(MODELS[j].__name__),feat,'train',str(avg_err[2,j])])

if __name__ == '__main__':
  main()
