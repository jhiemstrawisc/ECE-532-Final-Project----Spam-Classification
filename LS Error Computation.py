"""
Check errors
"""
import numpy as np
import winsound
from scipy.stats import binom

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second

winsound.Beep(frequency, duration)
print("Loading y")
y = np.loadtxt('ReducedLabels.csv')
print("Loading w_hat")
w_hat = np.loadtxt('weights.csv')
print("Loading A")
A = np.loadtxt('ReducedData.csv',delimiter=",",usecols=range(45849))
winsound.Beep(frequency, duration)
y_hat = A@w_hat

print(y_hat.shape,y.shape)
print(y_hat[0:100])

for index,i in enumerate(y_hat):
    if i>0:
        y_hat[index] = 1
    if i < 0:
        y_hat[index] = -1

print(y_hat[0:20])
misclass = 0 #number of misclassifications
false_pos = 0 #number of hams classified as spam
false_neg = 0 #number of spams classified as ham
rows = np.shape(y)
for i in range(rows[0]):
    if y_hat[i] != y[i]:
        misclass +=1
    if y_hat[i] == -1 and y[i] == 1:
        false_pos += 1
    if y_hat[i] == 1 and y[i] == -1:
        false_neg +=1
hams = 0
spams = 0
for i in range(rows[0]):
    if y[i] == 1:
        hams += 1
    elif y[i] == -1:
        spams += 1
false_neg    


print("Of the data used in this data set, there were " + str(hams) + " hams and " + str(spams) + " spams.")
percent_misclass = float(misclass)/rows[0] * 100
print("Percent of emails misclassified is " + str(round(percent_misclass,3)))
print("There were a total number of " + str(false_pos) + " false positives -- ie hams classified as spams.")
print("There were a total number of " + str(false_neg) + " false negatives -- ie spams that made it through.")
print("This means " + str(round(float(false_neg)/spams * 100,3)) + " percent of all spams made it through the detector.")
print("Of all the misclassifications, " + str(round(float(false_pos)/misclass * 100,3)) + " percent were false positives.")
print("Of all the misclassifications, " + str(round(float(false_neg)/misclass * 100,3)) + " percent were false negatives.")
ls_error = np.linalg.norm(y_hat-y,2)**2
print("The least squares error is " + str(ls_error))

print("If we assume a model that randomly guessed based on the observed distribution of spam/ham, \n then this model would follow the binomial distribution. As such, \n we can use the binomial cumulative mass function to gain some insight as to whether \n or not our model is better than guessing.")
print("Binomial CDF for 4588 trials (total emails) with a success rate of 0.682 (hams/spams) and 3127 successes (\#hams):")
print("The likelihood for doing better than we did by guessing is ",str(binom.cdf(k=4588, n=4588, p=0.681) - binom.cdf(k=3127, n=4588, p=0.681)))



