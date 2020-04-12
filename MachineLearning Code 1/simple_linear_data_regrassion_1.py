"""
Created on Sun Apr 12 22:42:59 2020
@author: mukul
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#splitting Data according to need
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#Predicting Observation
Y_pred = regressor.predict(X_test)
Y_pred_X_test = regressor.predict(X_train)

#visualizing the data
plt.scatter(X_train, Y_train , color='red')
plt.plot(X_train , Y_pred_X_test , color='blue')
plt.title('Slary vs experience (Trainig Set)')
plt.xlabel('Years of experience')
plt.ylabel('Slary')
plt.show()

#visualizing the data
plt.scatter(X_test, Y_test , color='red')
plt.plot(X_train , Y_pred_X_test , color='blue')
plt.title('Slary vs experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Slary')
plt.show()

