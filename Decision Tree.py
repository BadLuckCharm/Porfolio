import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix, classification_report, mean_squared_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Load data set
churn = pd.read_excel(r'F:\Masters\D209\Task 2\Prepared Data.xlsx')
#Split data set
y = churn['Churn']
x = churn.drop('Churn', axis=1)

#Finding best ccp_alpha for model
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=101)
clf_dt = DecisionTreeClassifier(random_state=101)
clf_dt = clf_dt.fit(x_train, y_train)
path = clf_dt.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[: -1]
clf_dts = []

alpha_loop_values = []
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=101, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, x_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.to_excel(r'F:\Masters\D209\Task 2\Alpha Results.xlsx', sheet_name='Alpha Results', index=False)
#alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')

ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.000780) & (alpha_results['alpha'] < 0.001097)]['alpha']
ideal_ccp_alpha = float(ideal_ccp_alpha)
ideal_ccp_alpha #0.0008789571430261288

#Model - Classification Tree - Confusion Matrix
clf_dt = DecisionTreeClassifier(random_state=101, ccp_alpha=ideal_ccp_alpha)
clf_dt = clf_dt.fit(x_train, y_train)
plt.figure(figsize=(40.96, 21.6))
plot_confusion_matrix(clf_dt, x_test, y_test, display_labels=["Active", "Disconnected"])
plt.show()
plt.figure(figsize=(40.96, 21.6))
plot_tree(clf_dt, filled=True, rounded=True, class_names=["Active", "Disconnected"], feature_names=x.columns)
plt.show()

extracted_MSEs = clf_dt.tree_.impurity
for idx, MSE in enumerate(clf_dt.tree_.impurity):
    print("Node {} has MSE {}".format(idx, MSE))



