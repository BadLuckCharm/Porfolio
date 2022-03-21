import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Load the CSV into python
churn = pd.read_csv(r'F:\Masters\D208\churn_clean.csv')

#Remove columns that are not needed/related to question
churn.drop(['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City',
            'State', 'County', 'Zip', 'TimeZone', 'Lat',
            'Lng', 'Population'], axis=1, inplace=True)

#Seperate the variables into categorical and continuous
categorical = ['Area', 'Contract', 'DeviceProtection', 'Gender', 'InternetService',
                     'Marital', 'Multiple', 'OnlineBackup', 'OnlineSecurity',
                     'PaperlessBilling', 'PaymentMethod', 'Phone', 'Port_modem',
                     'StreamingMovies', 'StreamingTV', 'Tablet', 'TechSupport', 'Techie']

continuous = ['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email',
                    'Contacts', 'Yearly_equip_failure', 'Tenure', 'MonthlyCharge',
                    'Bandwidth_GB_Year', 'Item1', 'Item2', 'Item3', 'Item4',
                    'Item5', 'Item6', 'Item7', 'Item8']

#Check for null values
print(churn.isnull().sum())

#Find number of unique values in each ordinal categorical column
for column in churn:
    unique_values = np.unique(churn[column])
    nbr_values = len(unique_values)
    if nbr_values < 10:
        print("The number of unique values in the column titled {} is: {} -- {}".format(column, nbr_values, unique_values))
    else:
        print("The number of unique values in the column titled {} is: {}".format(column, nbr_values))

#Remove columns with more than 10 unique values
churn.drop(['Job'], axis=1, inplace=True)

#Check for outliers
print(churn.describe())
sns.pairplot(churn[['Children', 'Outage_sec_perweek']])
plt.show()

#Find the ratio of active to disconnected customers
active = churn[churn['Churn'] == "Yes"]
disconnected = churn[churn['Churn'] == "No"]
active_pct = len(active)/len(churn['Churn'])
disconnected_pct = 1-active_pct

print('Total customers: ', len(churn))
print('Number of customers who have active services: ', len(active))
print('Number of customer who disconnected services: ', len(disconnected))
print('Active customers make up ', active_pct*100, '% of the data')
print('Disconnected customers make up ', disconnected_pct*100, '% of the data')

#Check relationship of target variable to each categorical variable
for c in categorical:
    sns.countplot(x=c, data=churn, palette='Set3', hue='Churn')
    plt.show()

#Get dummy values and convert target variable to numeric
churn1 = pd.get_dummies(churn, columns=categorical)
churn1['Churn'][churn1['Churn'] == 'Yes'] = 1
churn1['Churn'][churn1['Churn'] == 'No'] = 0

#Create target variable as y and predictor variables as x
y = churn1['Churn']
x = churn1.drop('Churn', axis=1).values
y = y.astype(int)

#Select initial predictor values
dt = DecisionTreeClassifier(random_state=15, criterion='entropy', max_depth=10)
dt.fit(x, y)
fi_col = []
fi = []
for i, column in enumerate(churn1.drop('Churn', axis=1)):
    print('The predictor value feature importance for {} is: {}'.format(column, dt.feature_importances_[i]))
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])
fi_df = zip(fi_col, fi)
fi_df = pd.DataFrame(fi_df, columns=['Feature Predictor', 'Feature Importance'])
fi_df = fi_df.sort_values('Feature Importance', ascending=False).reset_index()
initial_predictors = fi_df['Feature Predictor'][0:55]

#Export prepared data to
prepared_data = pd.concat([churn1['Churn'], churn1[initial_predictors]], axis=1)
prepared_data.to_excel(r'F:\Masters\D208\D208 Task 2\Prepared Data.xlsx', sheet_name='Prepared Data', index=False)
