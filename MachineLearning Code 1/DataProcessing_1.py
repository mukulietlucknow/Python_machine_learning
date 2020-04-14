import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

# Importing the dataset
dataset = pd.read_csv('Data.csv')
#splitting Data according to need
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#handling missing data
from sklearn.impute import SimpleImputer
imputer  = SimpleImputer(missing_values = np.nan , strategy='mean')
imputer.fit(X[: , 1:3])
X[: , 1:3] = imputer.transform(X[: , 1:3])

Y = LabelEncoder().fit_transform(Y) # Encoding Y data
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()