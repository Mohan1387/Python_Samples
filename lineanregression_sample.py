#importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

#conver boston data to data frame and check
boston1 = load_boston()
bos = pd.DataFrame(boston1.data)
print(bos.head())

#add column names
bos.columns = boston1.feature_names

#setting target
bos['PRICE'] = boston1.target
print(bos.head())

#loading the data in python and seperare dependent and independent variables

#independent variables
x = bos.drop('PRICE', axis = 1)

#dependent variable
y = bos['PRICE']

#dividing the dataset to tarining set and test set
#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.33, random_state = 5)

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(x, y, test_size = 0.33, random_state = 5)

#creating the Linear regression model
lm = LinearRegression()

#fit the model on train dataset
lm.fit(x_train, y_train)

#make predictions using created model
y_train_pred = lm.predict(x_train)

#use the model to predice the test dataset
y_test_pred = lm.predict(x_test)

#put the predicted and actual side by side to check the efficiency of the model
df = pd.DataFrame(y_test_pred,y_test)

print(df)


#calculate the MSE
from sklearn import metrics

mse = metrics.mean_squared_error(y_test, y_test_pred)
print(mse)


#plot and see where the model fits.
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label = 'Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='lightgreen', marker='o', label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y=0, xmin=0, xmax=50)
plt.plot()
plt.show()