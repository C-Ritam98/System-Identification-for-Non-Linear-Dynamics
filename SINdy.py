# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 07:29:37 2021

@author: KUNAL
"""

import numpy as np
import pysindy as ps
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
# First column is x, second is y
dataset=pd.read_csv("gravityoff.csv")
data=dataset.iloc[:,5:].values
time = dataset.iloc[:,0].values/1000000000

t=time[2000:3000]
X=data[2000:3000,:]
X=X.astype('float64')
poly_order = 3
threshold = 0.5
model1 = ps.SINDy(
    optimizer=ps.STLSQ(threshold=2,fit_intercept=True),
    feature_library=ps.PolynomialLibrary(degree=poly_order))
model2 = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.2,fit_intercept=False),
    feature_library=ps.FourierLibrary(n_frequencies=poly_order))
#model = ps.SINDy()
#model.fit(X, t=t)
model1.fit(X, t=t)
model2.fit(X, t=t)

# Model Parameter of both the model in terms of y` = f(y)

model1.print()
model2.print()

# testing

te=time[2000:2200]
X_test=data[2000:2200,:]
X_test=X_test.astype('float64')
#X_pred=model.simulate(X_test[0],te)
X_pred1=model1.simulate(X_test[0],te)
#X_pred2=model2.simulate(X_test[0],te)
#plt.scatter(te,X_pred2[:,48],color='r')

for i in [18,20,21,23,25,27,44,46,48,50]:
    print(mean_squared_error(X_test[:,50],X_pred1[:,50]))
    plt.scatter(te,X_test[:,i],label='X_test')
    plt.scatter(te,X_pred1[:,i],color='y',label='X_pred')
    plt.legend()
    plt.show()
#model.predict(np.array([[]]))
