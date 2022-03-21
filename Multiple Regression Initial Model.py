import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

churn = pd.read_excel(r'F:\Masters\D208\D208 Task 1\Prepared_Data.xlsx')

X = churn.drop(['Tenure'], axis=1)
y = churn['Tenure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression(fit_intercept=True)
regressor.fit(X_train, y_train)
regressor.fit(X_train, y_train)

print('Linear Model Coeff (m)', regressor.coef_)
print('Linear Model Coeff (b)', regressor.intercept_)

y_predict = regressor.predict(X_test)
plt.scatter(y_test, y_predict, color = 'r')
plt.ylabel('Model Predictions')
plt.xlabel('True (ground truth)')
plt.show()

k = X_test.shape[1]
n = len(X_test)

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)), '.3f'))
MSE = mean_squared_error(y_test,  y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1 - (1-r2) * (n-1) / (n-k-1)
MAPE =  np.mean(np.abs((y_test - y_predict) / y_test)) * 100

print('RMSE =', RMSE, '\nMSE =', MSE, '\nr2 =', r2, '\nadj_r2 =', adj_r2, '\nMAPE =', MAPE, '%')
