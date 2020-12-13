"""
More Preprocessing to make A non-singular (deleting duplicate emails)
"""
from scipy import linalg
import numpy as np

print("Loading y")
y = np.loadtxt('labels.csv')
print("Loading A")
A = np.loadtxt('DataMatrix.csv',delimiter=",",usecols=range(45849))
print("Phase 1")
rows,cols = np.shape(A)
counter = 0
indeces = []
for i in range(rows):
    indeces.append(counter)
    counter +=1 
deletes = []
print("Phase 2")
for i in range(rows):
    print(i)
    for j in indeces:
        if np.array_equiv(A[i,:],A[j,:]) and i != j:
            deletes.append(j)
            #A = np.delete(A,(j),axis=0)
            #y = np.delete(y, (j),axis = 0)
    indeces.pop(0)
print(deletes)
print("Phase 3")

deletes = np.unique(deletes)[::-1]
print(deletes)
for index in deletes:
    print(index)
    A = np.delete(A,(index),axis=0)
    y = np.delete(y, (index),axis = 0)

print("starting to write")
with open("ReducedData.csv",'w') as d:
    for row in A:
        for datum in row:
            d.write(str(datum) + ",")
        d.write('\n')
with open("ReducedLabels.csv",'w') as l:
    for datum in y:
        l.write(str(datum))
        l.write('\n')

"""
print("Starting the inverse... :/")
w_hat = A.T@linalg.inv(A@A.T)@y


#for i in w_hat:
    #print(i)
with open("weights.csv",'w') as w:
    for weight in w_hat:
        w.write(str(weight))
        w.write("\n")

"""
