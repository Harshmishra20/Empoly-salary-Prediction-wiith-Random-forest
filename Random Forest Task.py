# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 00:19:49 2023

@author: Dell
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'D:\Data Science\Daily Practice\March\21-03-2023\EMP SAL.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = y_train.reshape(-1, 1)
y_train = sc_y.fit_transform(y_train)

from sklearn.ensemble import RandomForestRegressor

regressor= RandomForestRegressor(n_estimators=10,criterion="poisson")
regressor.fit(x,y)

y_pred=regressor.predict([[6.5]])
y_pred1=regressor.predict([[6.5]])
y_pred2=regressor.predict([[6.5]])

y_pred3=regressor.predict([[6.5]])
y_pred4=regressor.predict([[6.5]])
y_pred5=regressor.predict([[6.5]])
y_pred6=regressor.predict([[6.5]])
y_pred7=regressor.predict([[6.5]])


y_pred8=regressor.predict([[6.5]])
y_pred9=regressor.predict([[6.5]])
y_pred10=regressor.predict([[6.5]])


y_pred11=regressor.predict([[6.5]])



x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x, y, color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()












