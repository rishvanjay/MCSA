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
timeTrain = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/EM_04062018/DATA00.csv'), dtype = float)
meanTrain = np.average(timeTrain)
time00 = timeTrain - meanTrain

def makeData(time):
    # welch transform
    x1, y1 = signal.welch(time, 10000, nperseg=65535/2)
    # decibel conversion
    y1=20*np.log10(y1)
    # give offset
    y1 = y1 + 100
    # find max and divide by max, then square the value to reduce noise
    # maxAmp = max(y1)
    # y2 = y1/maxAmp
    # y2 = np.square(y2)
    # y2 = y2*maxAmp
    # global frequency
    # frequency = x1
    return x1, y1

xx1, yy1 = makeData(time00)

freq = xx1.tolist()
amp = yy1.tolist()


def takeClosest(frequency):
    closeFreq = min(freq,key=lambda x:abs(x-float(frequency)))
    return closeFreq, freq.index(closeFreq)


desFreq = raw_input("Give Frequency: ")

number, index = takeClosest(desFreq)

print number, "   ", amp[index]

plt.plot(xx1, yy1, label = "EM 00")
plt.xlim(freq[index] - 25, freq[index] + 25)
plt.show()

