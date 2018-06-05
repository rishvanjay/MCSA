import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


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


#make test array
timeTest = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/newData/DATA01.csv'), dtype = float)
meanTest = np.average(timeTest)
time01 = timeTest - meanTest

timeTrain = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/newData/DATA02.csv'), dtype = float)
meanTrain = np.average(timeTrain)
time02 = timeTrain - meanTrain


#make test array
timeTest = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/Prestige Data/DATA00.csv'), dtype = float)
meanTest = np.average(timeTest)
time03 = timeTest - meanTest



timeTrain = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/Prestige Data/DATA01.csv'), dtype = float)
meanTrain = np.average(timeTrain)
time05 = timeTrain - meanTrain


#make test array
timeTest = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/Prestige Data/DATA02.csv'), dtype = float)
meanTest = np.average(timeTest)
time06 = timeTest - meanTest

# timeTest = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/newData/DATA07.csv'), dtype = float)
# meanTest = np.average(timeTest)
# time07 = timeTest - meanTest



# timeTrain = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/newData/DATA08.csv'), dtype = float)
# meanTrain = np.average(timeTrain)
# time08 = timeTrain - meanTrain


# #make test array
# timeTest = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/newData/DATA09.csv'), dtype = float)
# meanTest = np.average(timeTest)
# time09 = timeTest - meanTest

# timeTest = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/newData/DATA10.csv'), dtype = float)
# meanTest = np.average(timeTest)
# time10 = timeTest - meanTest

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

xx1, yy1 = makeData(time00)
xx2, yy2 = makeData(time01)
xx3, yy3 = makeData(time02)
xx4, yy4 = makeData(time03)
xx5, yy5 = makeData(time05)
xx6, yy6 = makeData(time06)
# xx7, yy7 = makeData(time07)
# xx8, yy8 = makeData(time08)
# xx9, yy9 = makeData(time09)
# xx10, yy10 = makeData(time10)

# for index, val in yy1:
#     if abs(yy1[index] - yy2[index]) > 0.05:
#         print xx1[index]


plt.subplot(211)

plt.plot(xx1, yy1, label = "EM 00")
plt.plot(xx2, yy2, label = "EM 01")
plt.plot(xx3, yy3, label = "EM 02")

plt.xlim(0, 1000)
plt.ylim(0, 100)

plt.legend()


plt.subplot(212)

plt.plot(xx4, yy4, label = "Prestige 00")
plt.plot(xx5, yy5, label = "Prestige 01")
plt.plot(xx6, yy6, label = "Prestige 02")

plt.xlim(0, 1000)
plt.ylim(0, 100)
plt.legend()

# plt.subplot(313)

# plt.plot(xx7, yy7, label = "07")
# plt.plot(xx8, yy8, label = "08")
# plt.plot(xx9, yy9, label = "09")
# plt.plot(xx10, yy10, label = "10")

# plt.xlim(0, 1000)
# plt.ylim(0, 0.5)
# plt.legend()


plt.show()