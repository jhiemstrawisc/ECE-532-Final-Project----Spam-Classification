"""
Create a dictionary
"""
import os
import string

word_list = []
punctuation_list = ['`','~','!','@','#','$','%','^','&','*','(',')','-','_','=','+','[','{',']','}','\\','|',':',';','"',"'",'<',',','>','.','?','/','0','1','2','3','4','5','6','7','8','9']
directory = r'C:\Users\Justin\Documents\ECE 532 -- Machine Learning\Final Project\enron1\all'
for entry in os.scandir(directory):
    email = entry.path
    
    with open(email,"r") as f:
        try:
            for line in f:      
                line = bytes(line, 'utf-8').decode("utf-8",'ignore')
                line_words = line.split()
                for word in line_words:
      
                    for character in word:
                        if character in punctuation_list:
                            word = word.replace(character,'')
                    if word not in word_list:
                        word_list.append(word)
        except:
            print(email)
with open("wordlist.txt", 'w') as w:
    for word in word_list:
        if word != '':
            w.write(word + '\n')






"""
huge_list = []

with open(, "r") as f:
    for line in f:
        huge_list.extend(line.split())
"""
