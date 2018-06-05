import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn import svm

# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score



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
timeTrain = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/Prestige Data/DATA00.csv'), dtype = float)
meanTrain = np.average(timeTrain)
timeTrain = timeTrain - meanTrain


#make test array
timeTest = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/Prestige Data/DATA05.csv'), dtype = float)
meanTest = np.average(timeTest)
timeTest = timeTest - meanTest

frequency = []
amp = []
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
    # global frequency
    # frequency = x1
    # global amp
    # amp = y1
    return np.vstack((x1, y2)).T, x1, y1

combinedTrain, f1, amp1 = makeData(timeTrain)
combinedTest, f2, amp2 = makeData(timeTest)


######################################################
#########  SVM MODEL  ################################
######################################################

clf = svm.OneClassSVM(nu=0.1, kernel="poly", gamma=0.1)
fit =  clf.fit(combinedTrain)
y_pred_train = clf.predict(combinedTrain)
y_pred_test = clf.predict(combinedTest)

print y_pred_test
print y_pred_train

count = 0

for index, value in enumerate(y_pred_train):
    if y_pred_test[index] != y_pred_train[index] and f1[index] < 1000:
        print str(f1[index]) + "    " + str(amp1[index]) 
        print str(f2[index]) + "    " + str(amp2[index]) 
        print y_pred_test[index]
        print y_pred_train[index]
        count+=1

print str(y_pred_test[165]), "   " , str(f2[165])



# plt.scatter(f1, amp1)
# plt.title("Welch Transform Prestige 00")


# w = clf.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(-5, 5)
# yy = a * xx - (clf.intercept_[0]) / w[1]

# plt.plot(xx, yy, 'k-')

# plt.show()