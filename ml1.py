import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'https://github.com/Yantra-Byte/Dataset/raw/main/Boston.csv')
df.columns
X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
X.shape
y = df['MEDV']
y.shape
#5 Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 252)
X_train
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.intercept_
model.coef_
y_pred = model.predict(X_test)
y_pred
y_test
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_pred)
