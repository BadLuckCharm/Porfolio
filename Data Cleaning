# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:44:25 2022

@author: Jsche
"""

import pandas as pd

#Load churn_raw_data.csv into python where column 0 is my index
churn = pd.read_csv('churn_raw_data.csv', index_col=0)
#Rename columns 43-51 with better naming convention
churn.rename({'item1': 'Survey_Timely_Response', 'item2': 'Survey_Timely_Fixes', 'item3': 'Survey_Timely_Replacements', 
              'item4': 'Survey_Reliability', 'item5': 'Survey_Options', 'item6': 'Survey_Respectful_Response',
              'item7': 'Survey_Courteous_Exchange', 'item8': 'Survey_Active_Listening'}, axis=1, inplace=True)
#Remove columns
churn.drop(['Lat', 'Lng', 'Interaction','Children', 'Age', 'Income', 'Techie', 'Phone', 'TechSupport', 'Tenure', 'Bandwidth_GB_Year', 'Zip'], axis=1, inplace=True)
#Remove all rows where there is NaN values (Removed for now, can reinstitute at a later time)
#churn.dropna(subset=['Children', 'Age', 'Income', 'Techie', 'Phone', 'TechSupport', 'Tenure', 'Bandwidth_GB_Year'], inplace=True)

churn.to_excel(r'C:\Users\Jsche\Desktop\Masters\D206\\Updated_churn_raw_data.xlsx', index=False)






