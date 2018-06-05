import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

 
magnitude = []
frequency = []
boolSpike = False

def read_csv(file):
    columns = defaultdict(list) # each value in each column is appended to a list

    with open(file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value 
                columns[k].append(v) # append the value into the appropriate list
                                    # based on column name k

    global magnitude
    magnitude = columns['Magnitude']
    global frequency
    frequency = columns['Frequency ']


read_csv('/Users/Rishvanjay/Desktop/Em_ML/MCSA_1.1_B_1.csv')

# half the arrays
frequency = frequency[:len(frequency)/2]
magnitude = magnitude[:len(magnitude)/2]

# index and values of array, converting string to float
for index, item in enumerate(frequency):
    frequency[index] = float(item)

for index, item in enumerate(magnitude):
    magnitude[index] = float(item)

# combining arrays in form [freq, mag]
mag = np.array(magnitude)
fre = np.array(frequency)
combined = np.vstack((fre, mag)).T

# returns index of closest frequency
def takeClosest(freq):
    closeFreq = min(frequency,key=lambda x:abs(x-freq))
    return frequency.index(closeFreq)

# boolean check if spike exists at given frequency
def check_spike(freq):
    index = takeClosest(freq)
    if magnitude[index] > 1:
       return True
    else:
        return False
#         print "spike around", freq, "=", boolSpike


# check_spike(50.0)
# check_spike(290.0)



# for freq in frequency:
#     check_spike(freq)

target = np.full((1024), False, dtype=bool)

for index, item in enumerate(frequency):
    target[index] = check_spike(item)

# print target[80]

######################################################
##  REGRESSION MODEL  ################################
######################################################

# Split the data into training/testing sets
X_train = combined[:-20]
X_test = combined[-20:]

# Split the targets into training/testing sets
y_train = target[:-20]
y_test = target[-20:]
# y_test[5] = True
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)


# Make predictions using the testing set
y_pred = regr.predict(X_test)


# The coefficients
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
print len(X_test)
print len(y_test)
plt.scatter(X_train[:,0], y_train,  color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()