import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

X1_test = []
X2_test = []
Y_test = []

X1_train = []
X2_train = []
Y_train = []

def read_csv(file):
    columns = defaultdict(list) # each value in each column is appended to a list

    with open(file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value 
                columns[k].append(v) # append the value into the appropriate list
                                    # based on column name k
    global X1_test
    X1_test = columns['X1_test']
    global X2_test
    X2_test = columns['X2_test']
    global Y_test
    Y_test = columns['Y_test']
    global X1_train
    X1_train = columns['\xef\xbb\xbfX1_train']
    global X2_train
    X2_train = columns['X2_train']
    global Y_train
    Y_train = columns['Y_train']
    
    # print Y_test
    # print X1_test
    # print X2_test
    # print Y_train
    # print X1_train
    # print X2_train

read_csv('/Users/Rishvanjay/Desktop/Em_ML/Initial Data/Data_for_ML_dummy.csv')
def makeFloat(list):
    for index, item in enumerate(list):
        list[index] = float(item)

makeFloat(X1_test)
makeFloat(X1_train)
makeFloat(X2_test)
makeFloat(X2_train)
makeFloat(Y_test)
makeFloat(Y_train)


x1te = np.array(X1_test)
x1tr = np.array(X1_train)
x2te = np.array(X2_test)
x2tr = np.array(X2_train)
yte = np.array(Y_test)
ytr = np.array(Y_train)

combined_test = np.vstack((x1te, x2te)).T
combined_train = np.vstack((x1tr, x2tr)).T

######################################################
##  REGRESSION MODEL  ################################
######################################################

# Split the data into training/testing sets
X_train = combined_train
X_test = combined_test


# Split the targets into training/testing sets
y_train = ytr
y_test = yte

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
# print len(X_test)
# print len(y_test)
# plt.scatter(X_train[:,0], y_train,  color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()

# print yte
# print y_pred


for index, val in enumerate(y_pred):
    if val >= 0.1:
        print "Freq", X1_test[index], "       Amp", X2_test[index]

print regr.score(X_test, y_test)