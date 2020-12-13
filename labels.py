"""
Need to generate our labels
"""
import os

directory = r'C:\Users\Justin\Documents\ECE 532 -- Machine Learning\Final Project\enron1\all'
counter = 0
for entry in os.scandir(directory):
    counter +=1
labels = [0]*counter





counter = 0
for entry in os.scandir(directory):
    email = entry.path
    if "spam.txt" in email:
        labels[counter] = -1
        
    elif "ham.txt" in email:
        labels[counter] =1
    counter +=1 
with open("labels.csv",'w') as f:
    for datum in labels:
        f.write(str(datum))
        f.write("\n")
    f.close()
