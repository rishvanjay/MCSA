import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def read_csv(file):
    columns = defaultdict(list) # each value in each column is appended to a list

    with open(file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value 
                columns[k].append(v) # append the value into the appropriate list
                                    # based on column name k
    return columns['Time']

# make time array
timeTrain = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/newData/DATA00.csv'), dtype = float)
meanTrain = np.average(timeTrain)
time00 = timeTrain - meanTrain

EM_train = time00


timeTest = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/EM faulty /DATA00_50Hz.csv'), dtype = float)
meanTest = np.average(timeTest)
time03 = timeTest - meanTest

PR_test = time03



def makeData(time):
    # welch transform
    x1, y1 = signal.welch(time, 10000, nperseg=65535/2)
    # decibel conversion
    y1=20*np.log10(y1)
    # give offset
    y1 = y1 + 100
    # find max and divide by max, then square the value to reduce noise
    maxAmp = max(y1)
    y2 = y1/maxAmp
    y2 = np.square(y2)
    y2 = y2*maxAmp
    # global frequency
    # frequency = x1
    return x1, y2

freq_train, amp_train = makeData(EM_train)

X_Train = np.vstack((freq_train, amp_train)).T

freq_test, amp_test = makeData(PR_test)

X_Test = np.vstack((freq_test, amp_test)).T

y_train = np.full((len(amp_train)), False, dtype=bool)
y_test = np.full((len(amp_test)), False, dtype=bool)


for index, val in enumerate(amp_train):
    if val > 8:
        y_train[index] = True
        # print "train  ", index
    else:
        y_train[index] = False

for index, val in enumerate(amp_test):
    if val > 8:
        y_test[index] = True
        # print "test  ", index
    else:
        y_test[index] = False





clf = SVC(kernel="linear", degree=2, verbose=True, tol=0.001)

print clf.fit(X_Train, y_train)
y_pred = clf.predict(X_Test)
print y_pred
count = 0
# for index, val in enumerate(y_pred):
#     if val != y_train[index] and index < 3000:
#         print freq_test[index]
#         print "predict  ", index
#         count = count + 1

print count
print clf.coef_
print clf.intercept_
print clf.support_vectors_

print X_Train[0]
np.put(X_Train[:, 0], 0, [30])
print X_Train[0]

