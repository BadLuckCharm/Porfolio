# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:47:55 2022

@author: Jsche
"""
import pandas as pd
import missingno as msno

#Load churn_raw_data.csv into python where column 0 is my index, labeled churn
churn = pd.read_csv('churn_raw_data.csv', index_col=0)
#Get information regarding the dataset, find any issues with data types not matching the column name
churn.info()
#Find duplicates in Customer_id
CID_Dupes = churn.duplicated('Customer_id',False)
Sorted_CID_Dupes = churn[CID_Dupes].sort_values('Customer_id')
Sorted_CID_Dupes[['Customer_id']]
#Find duplicates in Interaction to see if unique like Customer_id
Int_Dupes = churn.duplicated('Interaction',False)
Sorted_Int_Dupes = churn[Int_Dupes].sort_values('Interaction')
Sorted_Int_Dupes[['Interaction']]
#Create a list of columns that have NaN values
NaN_List = churn.columns[churn.isnull().any()].tolist()
#List of columns with NaN values
NaN_List = churn.filter(['Children', 'Age', 'Income', 'Techie', 'Phone', 'TechSupport', 'Tenure', 'Bandwidth_GB_Year'])
#Find type of dtype for each column that has missing values
NaN_List.info()
#Get histograms of colums with floats to find if there are any anomolies
NaN_List.hist()
#Check for randomness of missing data
sorted_children = churn.sort_values('Children')
msno.matrix(sorted_children)
sorted_age = churn.sort_values('Age')
msno.matrix(sorted_age)
sorted_income = churn.sort_values('Income')
msno.matrix(sorted_income)
sorted_techie = churn.sort_values('Techie')
msno.matrix(sorted_techie)
sorted_phone = churn.sort_values('Phone')
msno.matrix(sorted_phone)
sorted_techsupport = churn.sort_values('TechSupport')
msno.matrix(sorted_techsupport)
sorted_tenure = churn.sort_values('Tenure')
msno.matrix(sorted_tenure)
sorted_bandwidthgbyear = churn.sort_values('Bandwidth_GB_Year')
msno.matrix(sorted_bandwidthgbyear)
msno.heatmap(churn)
#Make sure there are no values equal to zero
churn['Bandwidth_GB_Year'].describe()
#Get unique values from string columns with NaN values
churn['Techie'].unique()
churn['Phone'].unique()
churn['TechSupport'].unique()
##Get unique values of the remaining integer data
#Check Zip Code to make sure digits are 5 digits long exactly and how many aren't
churn['Zip'].describe()
churn['Zip'][churn['Zip'] < 10000].count()
#Check Population to make sure is above 0 and not larger than possible
churn['Population'].describe()
#Check Outage in seconds to make sure the number is not too high and not negative
churn['Outage_sec_perweek'].describe()
#Check range of emails sent to customer
churn['Email'].describe()
#Check range of times customer contacted support
churn['Contacts'].describe()
#Check range of times customers equipment failed
churn['Yearly_equip_failure'].describe()
#Check range monthly charge
churn['MonthlyCharge'].describe()
#Check survey results
for i in range(8):
    churn[f'item{i + 1}'].describe()
