from frame_mod import *

# a dummy function for now that labels everything 1
def learning_method1(data, labels):
    return (lambda x: [1 for y in range(len(x))])

# Clean and crop images
folder1 = 'folder1'
folder2 = 'folder2'
(train_data,train_labels,test_data,test_labels) = import_data(folder1, folder2)

# run learning methods
# model should be a function that takes Image objects
model = learning_method1(train_data,train_labels)

# model should return an array of labels the same size as the test data
predictions = model(test_data)
print(predictions)

# compute error
sum_error = 0
for i in range(len(test_data)):
    if predictions[i]!=test_labels:
        sum_error+=1

error = sum_error/len(test_data)
print("Error is " + str(error))
