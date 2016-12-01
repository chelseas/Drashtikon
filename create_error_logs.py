import csv

filenames = ['error_hog2.csv', 'error_bright2.csv', 'error_random2.csv']
# diseases,model,feature,eval_set,error,,,
for this_file in filenames:
    with open(this_file,'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Diseases','Model','Features','Eval Set','Error Value']) #'Disease 1 Test Error','Disease 2 Test Error','Train Error'])
