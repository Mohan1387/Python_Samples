#############
#reference URL
#https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/
#
###########

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import sklearn
from sklearn.linear_model import LinearRegression
%matplotlib inline

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

#independent variables
x = bos.drop('PRICE', axis = 1)

#dependent variable
y = bos['PRICE']

#Scale X
x = scale(x)

#Initialize PCA for calculating 13 variables
pca = PCA(n_components=13)

#calculate PCA with dependent 13 columns
pca.fit(X)

#The amount of variance that each PC explains
 var= pca.explained_variance_ratio_

#Cumulative Variance explains
 var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
 print var1

#from the above output we can see that 9 varaliables defines 95% of data out of 13

#repeating the column drop like we did for target drop
#using Capital letter X and Y

X = bos.drop('PRICE', axis = 1)

#removing unwanted columns

X = bos.drop('TAX', axis = 1)
X = bos.drop('PTRATIO', axis = 1)
X = bos.drop('B', axis = 1)
X = bos.drop('LSTAT', axis = 1)

#dependent variable
Y = bos['PRICE']

#dividing the dataset to tarining set and test set
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)

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

df = pd.DataFrame(y_train_pred,y_train)

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
