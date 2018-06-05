print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# print diabetes_y_pred
# The coefficients
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

















# from sklearn import datasets
# from sklearn import linear_model
# import numpy as np
# import matplotlib.pyplot as plt 

# diabetes = datasets.load_diabetes()
# diabetes_X_train = diabetes.data[:-20]
# diabetes_X_test  = diabetes.data[-20:]
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test  = diabetes.target[-20:]
# print dir(datasets.load_diabetes())

# regr = linear_model.LinearRegression()
# print regr.fit(diabetes_X_train, diabetes_y_train)
# # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# print np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)
# print regr.score(diabetes_X_test, diabetes_y_test) 

# print len(diabetes.data)

# plt.scatter(diabetes_X_test, diabetes_y_test, color ='black')  

# plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3) 
# plt.xticks(())
# plt.yticks(())
# plt.show()

# # iris = datasets.load_iris()
# # iris_X = iris.data
# # iris_y = iris.target
# # np.unique(iris_y)

# # print iris_X
# # print iris_y

