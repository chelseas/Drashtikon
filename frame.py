from frame_mod import *
from SVM import *
import numpy
from skimage.feature import hog

# a dummy function for now that labels everything 1
def learning_method1(data, labels):
    return (lambda x: [1 for y in range(len(x))])

# Clean and crop images
folder1 = 'folder1'
folder2 = 'folder2'
(train_data,train_labels,test_data,test_labels) = import_data(folder1, folder2)

# run learning methods
# model should be a function that takes Image objects
print(size(numpy.array(train_data[0])))
print(type(train_data[0]))
#features = [numpy.ndarray.tolist(hog(numpy.ndarray(x))) for x in train_data]
features = getHogFeatures(train_data)
model = SVM(features,train_labels)

# model should return an array of labels the same size as the test data
predictions = model(getHogFeatures(test_data))
print(predictions)

# compute error
sum_error = 0
for i in range(len(test_data)):
    if predictions[i]!=test_labels[i]:
        sum_error+=1

error = sum_error/len(test_data)
print("Error is " + str(error))

## TODO: print to textfile
