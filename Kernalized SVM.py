import numpy as np
from sklearn import svm
import datetime
from random import *

print("Loading y")
y = np.loadtxt('Labels.csv')

    
print("Loading A")
A = np.loadtxt('DataMatrix.csv',delimiter=",",usecols=range(45849))

rows = y.shape

validation_indeces = sample(range(rows[0]), int(rows[0]/3))
training_indeces = [index for index in range(rows[0]) if index not in validation_indeces]

training_matrix = np.array([A[i,:] for i in training_indeces])
training_y = np.array([y[i] for i in training_indeces])

validation_matrix = np.array([A[i,:] for i in validation_indeces])
validation_y = np.array([y[i] for i in validation_indeces])


def KernalizedSVM(ker,reg,cache,deg):
    #classifier = svm.SVC(kernel=ker,C=reg, cache_size=cache, degree = deg) #For standard kernelized SVM
    classifier = svm.NuSVC(kernel=ker,cache_size=cache, degree = deg) #For NuSVM
    print("starting SVM at " + str(datetime.datetime.now()))
    classifier.fit(training_matrix,training_y)
    hams = 0
    spams = 0
    misclass = 0
    false_pos = 0
    false_neg = 0
    for index,value in enumerate(validation_y):
        output = classifier.predict(validation_matrix[index,:].reshape(1,-1))
        if value == 1:
            hams +=1
        else:
            spams +=1
        if value != output[0]:
            misclass +=1
            if value == 1 and output[0] == -1:
                false_pos +=1
            else:
                false_neg += 1
   
    print("Of the data used to validate this neural net, there were ",hams," hams and ",spams," spams.")
    print("There were ",misclass," misclassifications.")
    print("This equates to a misclassification rate of ",round(float(misclass)/(hams+spams) * 100,3), " percent.")
    print("There were ",false_pos," false positives (ham incorrectly labeled spam) and ",false_neg," false negatives (spam labeled as ham).")
    print("This means ", round(false_neg/spams * 100,3), " percent of spams made it through the filter.")
    print("Of all the misclassifications, ",round(false_pos/misclass * 100,3)," percent were false positives.")
    print("Of all the misclassifications, ",round(false_neg/misclass * 100,3)," percent were false negatives.")
  
    with open(str(ker)+"-"+str(reg) + "-" +str(deg) +".txt",'w') as f:
        f.write("Of the data used to validate this neural net, there were " + str(hams) + " hams and " + str(spams) + " spams." + "\n")
        f.write("There were "+ str(misclass) + " misclassifications."+ "\n")
        f.write("This equates to a misclassification rate of " + str(round(float(misclass)/(hams+spams) * 100,3)) + " percent."+ "\n")
        f.write("There were " + str(false_pos) + " false positives (ham incorrectly labeled spam) and " + str(false_neg) + " false negatives (spam labeled as ham)."+ "\n")
        f.write("This means " + str(round(false_neg/spams * 100,3)) + " percent of spams made it through the filter."+ "\n")
        f.write("Of all the misclassifications, " + str(round(false_pos/misclass * 100,3)) + " percent were false positives."+ "\n")
        f.write("Of all the misclassifications, " + str(round(false_neg/misclass * 100,3)) + " percent were false negatives."+ "\n")
        f.close





KernalizedSVM("rbf",0.000001,1000,0)
KernalizedSVM("rbf",0.0001,1000,0)
KernalizedSVM("rbf",0.001,1000,0)
KernalizedSVM("rbf",0.1,1000,0)

