import openpyxl
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

# Load prepared data
churn = pd.read_excel(r'F:\Masters\D208\D208 Task 2\Prepared Data.xlsx')

# Create x and y variables
x = churn.drop('Churn', axis=1).values
y = churn['Churn']

# Create testing and training split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

#Training and predicting
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)

#Confusion matrix
y_pred = logmodel.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)

#Classification report
print("\n" "Classification Report:")
print(classification_report(y_test, predictions))
