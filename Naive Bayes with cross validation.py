"""
Partial Training Naive Bayes
"""

import numpy as np
from sklearn import svm
import datetime
from random import *


np.seterr(all='print')


print("Loading y")
y = np.loadtxt('Labels.csv')
 
print("Loading A")
A = np.loadtxt('DataMatrix.csv',delimiter=",",usecols=range(45849))

emails,words = A.shape

rows = y.shape

validation_indeces = sample(range(rows[0]), int(rows[0]/12))
training_indeces = [index for index in range(rows[0]) if index not in validation_indeces]

training_matrix = np.array([A[i,:] for i in training_indeces])
training_y = np.array([y[i] for i in training_indeces])

validation_matrix = np.array([A[i,:] for i in validation_indeces])
validation_y = np.array([y[i] for i in validation_indeces])

validation_rows = validation_y.shape



p_word_given_spam = [0]*words
p_word_given_ham = [0]*words
p_spam_email = 0
p_ham_email = 0
p_spam_given_word = [0]*words
p_ham_given_word = [0]*words

#We want as close to the underlying likelihood of spam, so we use the whole training set on this
total_spam = 0
total_ham = 0
for i in y:
    if i == -1:
        total_spam += 1
    else:
        total_ham +=1

p_spam = total_spam/emails
p_ham = total_ham/emails


spam_word_list = [0]*words
ham_word_list = [0]*words

for email_index,value in enumerate(training_y):
    if value == -1: #we're in a spam email
        for word_ind,count in enumerate(training_matrix[email_index,:]):
            spam_word_list[word_ind] += count
    elif value == 1:
        for word_ind,count in enumerate(training_matrix[email_index,:]):
            ham_word_list[word_ind] += count  


total_spam_words = sum(spam_word_list)
total_ham_words = sum(ham_word_list)
for index,value in enumerate(spam_word_list):
    p_word_given_spam[index] = value/total_spam_words
for index,value in enumerate(ham_word_list):
    p_word_given_ham[index] = value/total_ham_words
#print(p_word_given_ham[0])
#print(p_word_given_spam[0])



for i in range(words):
    if p_word_given_spam[i] !=0 and p_word_given_ham != 0:
        p_spam_given_word[i] = (p_word_given_spam[i]*p_spam)/(p_word_given_spam[i]*p_spam + p_word_given_ham[i]*p_ham)


"""
We now need to account for words that were not in the training set
Because these words will have a probability of 0 when we do later calculations
they cause problems. We're going to assume for now that if a word
hasn't been encountered before, P(S|W) = 0.5, ie we pass no judgements about
the likelihood of spam given that word. We first need to set any 0s
in p_spam_given_word to this value
"""

for i in range(words):
    if p_spam_given_word[i] == 0:
        p_spam_given_word[i] = 0.5
#for i in range(20):
    #print(p_spam_given_word[i])

for i in p_spam_given_word:
    if i == 0:
        print("you found a 0")


"""
We now have a vector containing P(s|W) for all our words. The next step is to look
at a suspect email and use the associate equation to determine the likelihood
that the email is spam
"""

y_hat = [0]*validation_rows[0]
for index in range(validation_rows[0]):
    numerator = 1
    part_denom = 1
    for word_ind,word_value in enumerate(validation_matrix[index,:]):
        if word_value != 0:
            numerator = numerator * p_spam_given_word[word_ind]**word_value
            part_denom = part_denom * (1 - p_spam_given_word[word_ind])**word_value
            #print(numerator,part_denom)
    p_email_is_spam = numerator/(numerator + part_denom)
    if p_email_is_spam >= 0.5:
        y_hat[index] = -1
    elif p_email_is_spam < 0.5:
        y_hat[index] = 1


misclass = 0
false_neg = 0
false_pos = 0
for index,value in enumerate(y_hat):
    if validation_y[index] != y_hat[index]:
        misclass +=1
    if validation_y[index] == 1 and y_hat[index] == -1:
        false_pos += 1
    elif validation_y[index] == -1 and y_hat[index] == 1:
        false_neg +=1


weird_things = 0
validation_hams = 0
validation_spams = 0
for i in validation_y:
    if i == 1.0:
        validation_hams +=1
    elif i == -1.0:
        validation_spams +=1
    elif i != 1 and i != -1:
        print(i)
        weird_things += 1

print("There were ",weird_things," weird things that happened.")
    
print("Of the data used to validate this neural net, there were ",validation_hams," hams and ",validation_spams," spams.")
print("There were ",misclass," misclassifications.")
print("This equates to a misclassification rate of ",round(float(misclass)/(validation_hams + validation_spams) * 100,3), " percent.")
print("There were ",false_pos," false positives (ham incorrectly labeled spam) and ",false_neg," false negatives (spam labeled as ham).")
print("This means ", round(false_neg/validation_spams * 100,3), " percent of spams made it through the filter.")
print("Of all the misclassifications, ",round(false_pos/misclass * 100,3)," percent were false positives.")
print("Of all the misclassifications, ",round(false_neg/misclass * 100,3)," percent were false negatives.")



























