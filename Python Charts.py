    # -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:14:10 2022

@author: Jsche
"""

import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

churn = pd.read_csv(r'C:\Users\Jsche\Desktop\Masters\D207\\churn_clean.csv')

ax = sns.boxplot(x = 'Contacts', y='Age', data=churn)
plt.show()

nocontact = churn[churn.Contacts == 0].Age
onecontact = churn[churn.Contacts == 1].Age
twocontact = churn[churn.Contacts == 2].Age
threecontact = churn[churn.Contacts == 3].Age
fourcontact = churn[churn.Contacts == 4].Age
fivecontact = churn[churn.Contacts == 5].Age
sixcontact = churn[churn.Contacts == 6].Age
sevencontact = churn[churn.Contacts == 7].Age

anova = stats.f_oneway(nocontact, onecontact, twocontact, threecontact, fourcontact, fivecontact, sixcontact, sevencontact)
print(anova)

