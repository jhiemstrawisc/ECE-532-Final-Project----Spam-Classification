import os

directory = r'C:\Users\Justin\Documents\ECE 532 -- Machine Learning\Final Project\enron1\all'
punctuation_list = ['`','~','!','@','#','$','%','^','&','*','(',')','-','_','=','+','[','{',']','}','\\','|',':',';','"',"'",'<',',','>','.','?','/','0','1','2','3','4','5','6','7','8','9']

word_list = []
with open("wordlist.txt",'r') as f:
    for line in f:
        word_list.append(line.rstrip())
print(len(word_list))
#print(word_list[0:200])

#we need to make a dictionary



for entry in os.scandir(directory):
    email = entry.path
    email_words = []
    email_vector = [0]*len(word_list)
    with open(email,"r") as f:
        for line in f:
            line = bytes(line, 'utf-8').decode("utf-8",'ignore')
            line_words = line.split()
            for word in line_words:
                    for character in word:
                        if character in punctuation_list:
                            word = word.replace(character,'')
                    if word not in email_words:
                        email_words.append(word)
    for word in word_list:
        if word in email_words:
            position = word_list.index(word)
            email_vector[position] += 1
    
    with open("DataMatrix.txt",'a') as d:
        for datum in email_vector:
            d.write(str(datum)+ ",")
        d.write('\n')
