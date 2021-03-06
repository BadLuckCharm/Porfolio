import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Load the CSV into python
churn = pd.read_csv(r'F:\Masters\D208\churn_clean.csv')

#Remove columns that I know will not be used
churn.drop(['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City',
            'State', 'County', 'Zip', 'TimeZone', 'Lat',
            'Lng', 'Population', 'Job', 'PaymentMethod', 'Port_modem',
            'Item1', 'Item2', 'Item3', 'Item4', 'Item5',
            'Item6', 'Item7', 'Item8'], axis=1, inplace=True)

# Assign dummy values to categorical categories
header_list: list = ['Churn', 'Techie', 'Tablet', 'Phone', 'Multiple',
                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                     'Area', 'Marital', 'Gender', 'Contract', 'InternetService']
churn = pd.get_dummies(churn, drop_first=True, columns=header_list)

#Create heatmap with correlation values to find any value above 0
k = 7
corr_matrix = churn.corr()
cols = corr_matrix.nlargest(k, 'Tenure')['Tenure'].index
cm = np.corrcoef(churn[cols].values.T)
sns.set(font_scale=.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Final list of variables
churn = churn[['Tenure', 'Bandwidth_GB_Year', 'OnlineBackup_Yes', 'Contract_Two Year',
               'Age', 'Yearly_equip_failure', 'Marital_Married']]

#Check for outliers in continuous variables
sns.jointplot(x='Bandwidth_GB_Year', y='Tenure', data=churn)
sns.jointplot(x='Age', y='Tenure', data=churn)
sns.jointplot(x='Yearly_equip_failure', y='Tenure', data=churn)
plt.show()

#Remove outlier data
churn.drop(churn[(churn['Yearly_equip_failure'] > 3)].index, inplace=True)

#Write prepared data to excel
churn.to_excel("F:\Masters\D208\D208 Task 1\Prepared_Data.xlsx", index=False)
