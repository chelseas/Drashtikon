from frame_mod import *
from SVM import *
import numpy
from skimage.feature import hog
import argparse

parser = argparse.ArgumentParser(description='Framework that trains all learning options.')
parser.add_argument('class1input' , action="store", help='Input Photo Directory, relative path to script directory')
parser.add_argument('class2input' , action="store", help='Input Photo Directory, relative path to script directory')
args = parser.parse_args()


def main():
  path = os.path.dirname(os.path.realpath(__file__))
  inputPath1 = os.path.join(path, args.class1input)
  inputPath2 = os.path.join(path, args.class2input)
  (train_data,train_labels,test_data,test_labels) = import_data(inputPath1, inputPath2)
  features = numpy.array([hog(x) for x in train_data])
  model = SVM(features,train_labels)
  predictions = model(numpy.array([hog(x) for x in test_data]))
  
  test_labels = numpy.array(test_labels)
  
  print(test_labels)
  print(predictions)

  ## TODO: print to textfile
  sum_error = 0
  for i in range(len(test_labels)):
    if predictions[i] != test_labels[i]:
        sum_error+=1
  error = float(sum_error)/float(len(test_data))
  print("Error is " + str(error))




if __name__ == '__main__':
  main()



