"""
Naive Bayes Filter
"""
import numpy as np
from sklearn import svm
import datetime
from random import *
print("Loading y")
y = np.loadtxt('Labels.csv')

    
print("Loading A")
A = np.loadtxt('DataMatrix.csv',delimiter=",",usecols=range(45849))

emails,words = A.shape
print(emails,words)
#emails = 5172 words = 45849


p_word_given_spam = [0]*words
p_word_given_ham = [0]*words
p_spam_email = 0
p_ham_email = 0
p_spam_given_word = [0]*words
p_ham_given_word = [0]*words


total_spam = 0
total_ham = 0
for i in y:
    if i == -1:
        total_spam += 1
    else:
        total_ham +=1

p_spam = total_spam/emails
p_ham = total_ham/emails


"""
To find P(W|S) we first need to collect a list of all the spam words
"""

spam_word_list = [0]*words
ham_word_list = [0]*words

for email_index,value in enumerate(y):
    if value == -1: #we're in a spam email
        for word_ind,count in enumerate(A[email_index,:]):
            spam_word_list[word_ind] += count
    elif value == 1:
        for word_ind,count in enumerate(A[email_index,:]):
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
    p_spam_given_word[i] = (p_word_given_spam[i]*p_spam)/(p_word_given_spam[i]*p_spam + p_word_given_ham[i]*p_ham)

#for i in range(20):
    #print(p_spam_given_word[i])




"""
We now have a vector containing P(s|W) for all our words. The next step is to look
at a suspect email and use the associate equation to determine the likelihood
that the email is spam
"""

y_hat = [0]*emails
for index in range(emails):
    numerator = 1
    part_denom = 1
    for word_ind,word_value in enumerate(A[index,:]):
        if word_value != 0:
            numerator = numerator * p_spam_given_word[word_ind]**word_value
            part_denom = part_denom * (1 - p_spam_given_word[word_ind])**word_value
    p_email_is_spam = numerator/(numerator + part_denom)
    if p_email_is_spam >= 0.5:
        y_hat[index] = -1
    elif p_email_is_spam < 0.5:
        y_hat[index] = 1


misclass = 0
false_neg = 0
false_pos = 0
for index,value in enumerate(y):
    if y[index] != y_hat[index]:
        misclass +=1
    if y[index] == 1 and y_hat[index] == -1:
        false_pos += 1
    elif y[index] == -1 and y_hat[index] == 1:
        false_neg +=1

print("Of the data used to validate this Naive Bayes Filter, there were ",total_ham," hams and ",total_spam," spams.")
print("There were ",misclass," misclassifications.")
print("This equates to a misclassification rate of ",round(float(misclass)/(total_ham+total_spam) * 100,3), " percent.")
print("There were ",false_pos," false positives (ham incorrectly labeled spam) and ",false_neg," false negatives (spam labeled as ham).")
print("This means ", round(false_neg/total_spam * 100,3), " percent of spams made it through the filter.")
print("Of all the misclassifications, ",round(false_pos/misclass * 100,3)," percent were false positives.")
print("Of all the misclassifications, ",round(false_neg/misclass * 100,3)," percent were false negatives.")

        















    
    
