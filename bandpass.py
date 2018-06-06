import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, fftpack
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

# timeTest = np.array(read_csv('/Users/Rishvanjay/Desktop/Em_ML/EM faulty /DATA00_50Hz.csv'), dtype = float)
# meanTest = np.average(timeTest)
# time00 = timeTest - meanTest

# plt.subplot(311)

# plt.plot(np.arange(0, 6.5535, 0.0001), fftpack.hilbert(time00))
# plt.subplot(312)
# plt.plot(np.arange(0, 6.5535, 0.0001), time00)


def makeData(time):
    
    # butterworth
    b, a = signal.butter(5, [0.002, 0.2], btype='bandpass', analog=False)

    zi = signal.lfilter_zi(b, a)
    filtered, z0 = signal.lfilter(b, a, time, zi=zi*time[0])

    
    
    # welch transform
    x1, y1 = signal.welch(filtered, 10000, nperseg=65535/2)
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
# xx2, yy2 = makeData(time01)

# plt.subplot(313)

plt.plot(xx1, yy1, label = "EM 00 Faulty without drive")

# plt.plot(xx2, yy2, label = "EM 00 Faulty with drive")
# plt.plot(xx3, yy3, label = "EM 02")

plt.xlim(0, 1000)
# plt.ylim(0, 100)

plt.legend()

plt.show()