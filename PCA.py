# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:06:49 2022

@author: Jsche
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA



#Load churn_raw_data.csv into python where column 0 is my index
churn = pd.read_csv('Updated_churn_raw_data.csv', index_col=0)

churn_pca = churn[['MonthlyCharge',
                   'Survey_Timely_Response', 'Survey_Timely_Fixes', 'Survey_Timely_Replacements',
                   'Survey_Reliability', 'Survey_Options', 'Survey_Respectful_Response', 
                   'Survey_Courteous_Exchange', 'Survey_Active_Listening']]

churn_pca_normalized = (churn_pca-churn_pca.mean())/churn_pca.std()

pca = PCA(n_components=churn_pca.shape[1])

pca.fit(churn_pca_normalized)

pca2 = pd.DataFrame(pca.transform(churn_pca_normalized),columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9'])

loadings = pd.DataFrame(pca.components_.T,
                        columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9'],
                        index=churn_pca_normalized.columns)




cov_matrix = np.dot(churn_pca_normalized.T, churn_pca_normalized) / churn_pca.shape[0]
eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]

plt.plot(eigenvalues)
plt.xlabel('number of components')
plt.ylabel('eigenvalues')
plt.show()

loadings = np.cumsum(loadings)
loadings.to_excel(r'C:\Users\Jsche\Desktop\Masters\D206\\loadings.xlsx', index=False)

