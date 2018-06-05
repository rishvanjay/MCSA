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


# # fft on time components
# ft = np.fft.fft(nptime)

# ft = np.absolute(ft)
# # db = 20 * np.log10(ft)

# # fft frequency component
# freq = np.fft.fftfreq(nptime.size, 0.0001)

# # welch 
# f, xx = signal.periodogram(nptime, 0.0001)



t = np.arange(0, 6.5535, 0.0001)

rand = np.random.rand(65536)
sin = np.sin(2*np.pi*50*t)
sin = sin + 0.1*np.sin(2*np.pi*97.5*t) + 0.1*np.sin(2*np.pi*2.5*t)
sin = sin + 0.8*np.sin(2*np.pi*55*t) + 0.8*np.sin(2*np.pi*45*t)
# sin = sin + 0.4*np.sin(2*np.pi*145*t) + 0.4*np.sin(2*np.pi*(-45)*t)
sin = sin + 0.7*np.sin(2*np.pi*60*t) + 0.7*np.sin(2*np.pi*40*t)
sin = sin + 0.6*np.sin(2*np.pi*70.6*t) + 0.6*np.sin(2*np.pi*29.4*t)
sin = sin + 0.3*np.sin(2*np.pi*79.4*t) + 0.3*np.sin(2*np.pi*20.6*t)
sin = sin + 0.5*np.sin(2*np.pi*63.75*t) + 0.5*np.sin(2*np.pi*36.25*t)
sin = sin + + np.random.rand(*sin.shape)


# mean = np.average(sin)

# sin = sin - mean

# sinfft = np.fft.fft(sin, norm='ortho')
# freq = np.fft.fftfreq(sinfft.size, 0.0001)
# sinfft=20*np.log10(sinfft) 

f, xx = signal.periodogram(sin, 10000)
xx=20*np.log10(xx)

f1, xx1 = signal.welch(sin, 10000, nperseg=65535/2)
xx1=20*np.log10(xx1)

frequency = f1.tolist()
mag = xx1 + 100
mag = mag.tolist()

###########################################

def check_spike(freq):
    index = frequency.index(value)
    if mag[index] > 30:
       return True
    else:
        return False

# print check_spike(50)

for index, value in enumerate(mag):
    if index > 1 and index < len(mag) - 1:
        if (mag[index] > mag[index+1]) and mag[index] > mag[index - 1] and mag[index] > 20:
            print frequency[index]

###########################################
################  PLOT  ###################

# print f

# plt.subplot(411)
# plt.plot(t, sin)
# plt.title("Sine wave")

# plt.subplot(412)
# plt.plot(freq, sinfft.real)
# plt.title("FFT")
# # plt.xlim((0,5000))
# # plt.ylim(-1, 5)

# plt.subplot(211)
# plt.plot(f, xx)
# plt.title("Periodogram")


# plt.subplot(212)

# remove noise

# xx1 = xx1+100

maxAmp = max(xx1)

xx2 = xx1/maxAmp

xx2 = np.square(xx2)

xx2 = xx2*maxAmp

plt.subplot(211)
plt.plot(f1, xx1)
plt.title("Welch Transform with noise")

plt.subplot(212)
plt.plot(f1, xx2)
plt.title("Welch Transform after removing noise")

# plt.xlim((0,2))
# plt.ylim(-0.2, 1.2)

# plt.psd(sin)

plt.show()