import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline

#Loading the prepared data
churn = pd.read_excel(r'F:\Masters\D209\Task 1\Prepared Data.xlsx')
churn_col = ['Churn', 'Tenure', 'MonthlyCharge', 'Contract_Month-to-month']

#Visualizations of the data
sns.pairplot(data=churn, hue='Churn')
plt.show()

#Setting x and y variables
x = churn.drop('Churn', axis=1)
y = churn['Churn']

#Setting up  x/y traininga and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Initial prediction with one N_neighbor
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(scaled_x_train, y_train)
y_pred = knn_model.predict(scaled_x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Selecting the best K value
test_error_rates = []
for k in range(1, 30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_x_train, y_train)
    y_pred_test = knn_model.predict(scaled_x_test)
    test_error = 1-accuracy_score(y_test, y_pred_test)
    test_error_rates.append(test_error)

print(test_error_rates)
plt.plot(range(1, 30), test_error_rates)
plt.ylabel('Error Rate')
plt.xlabel('K Neighbors')
plt.show()

#Pipeline to compare k values for optimal k value
scaler = StandardScaler()
knn = KNeighborsClassifier()
operations = [('scaler', scaler), ('knn', knn)]
pipe = Pipeline(operations)
k_values = list(range(1, 18))
param_grid = {'knn__n_neighbors': k_values}
full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
full_cv_classifier.fit(x_train, y_train)
full_cv_classifier.best_estimator_.get_params()

#Final predictions with 14 k values
full_pred = full_cv_classifier.predict(x_test)
print(classification_report(y_test, full_pred))

#Testing the data
customer1 = [[35, 179.99, 0]]
print(full_cv_classifier.predict_proba(customer1))

