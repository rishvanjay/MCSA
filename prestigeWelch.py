import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

time = []

def read_csv(file):
    columns = defaultdict(list) # each value in each column is appended to a list

    with open(file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value 
                columns[k].append(v) # append the value into the appropriate list
                                    # based on column name k
    global time
    time = columns['Time']



read_csv('/Users/Rishvanjay/Desktop/Em_ML/Prestige Data/DATA00.csv')

nptime = np.array(time, dtype = float)
mean = np.average(nptime)

nptime = nptime - mean
t = np.arange(0, 6.5535, 0.0001)

# butterworth
b, a = signal.butter(20, 0.4, 'low', analog=False)

zi = signal.lfilter_zi(b, a)
filtered, z0 = signal.lfilter(b, a, nptime, zi=zi*nptime[0])



f1, xx1 = signal.welch(filtered, 10000, nperseg=65535/2)
xx1=20*np.log10(xx1)

# xx1 = xx1+500


###########################################

# def check_spike(freq):
#     index = frequency.index(value)
#     if mag[index] > 30:
#        return True
#     else:
#         return False

# print check_spike(50)








frequency = f1.tolist()
mag = xx1 + 100

xx1 = xx1+100

maxAmp = max(xx1)

xx2 = xx1/maxAmp

xx2 = np.square(xx2)

xx2 = xx2*maxAmp
mag = xx2
mag = mag.tolist()

for index, value in enumerate(mag):
    if index > 1 and index < len(mag) - 1:
        if (mag[index] > mag[index+1]) and mag[index] > mag[index - 1] and mag[index] > 15 and frequency[index] < 2000:
            print str(frequency[index]) + "   " + str(mag[index])






plt.subplot(311)
plt.plot(t, filtered)
# plt.title("Welch Transform with noise")

plt.subplot(312)
plt.plot(f1, xx1)
# plt.title("Welch Transform after removing noise")


plt.subplot(313)
plt.plot(f1, xx2)
# plt.title("Welch Transform after removing noise")



plt.show()

