# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 23:04:28 2020

@author: DELL
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)

# fitting simple linear regression on the training test

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict test set result
 y_pred =regressor.predict(X_test)
 y_pred_train=regressor.predict(X_train)
 
# visualize test training tes
 
 plt.scatter(X_train,y_train,color='red')
 plt.plot(X_train, y_pred_train,color='blue')
 plt.title('Years of experience vs Salary')
 plt.xlabel('Years of experience')
 plt.ylabel('Salary')
 plt.show()
 
 
# visualize test test tes
 plt.scatter(X_test,y_test,color='red')
 plt.plot(X_train,regressor.predict(X_train),color='blue')
 plt.title('Years of experience vs Salary(test set)')
 plt.xlabel('Years of experience')
 plt.ylabel('Salary')
 plt.show()