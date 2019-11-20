import requests
import warnings
warnings.filterwarnings('ignore')
import json
from pprint import pprint


from os import path

import matplotlib
import matplotlib.pyplot as plt

from datetime import timedelta, datetime

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR

from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics

import math

# def evaluate(yhat, test, output=True):
#     mse = metrics.mean_squared_error(test, yhat)
#     rmse = math.sqrt(mse)
    
#     if output:
#         print('MSE: {}'.format(mse))
#         print('RMSE: {}'.format(rmse))
#     else:
#         return mse, rmse

def evaluate(yhat, target_var, train, test, output=True):
    mse = metrics.mean_squared_error(test[target_var], yhat[target_var])
    rmse = math.sqrt(mse)
    
    if output:
        print('MSE: {}'.format(mse))
        print('RMSE: {}'.format(rmse))
    else:
        return mse, rmse

def plot_and_eval(yhat, target_vars, train, test, metric_fmt = '{:.2f}', linewidth = 4):
    if type(target_vars) is not list:
        target_vars = [target_vars]

    plt.figure(figsize=(16, 8))
    plt.plot(train[target_vars],label='Train', linewidth=1)
    plt.plot(test[target_vars], label='Test', linewidth=1)

    for var in target_vars:
        mse, rmse = evaluate(yhat=yhat, target_var = var, train = train, test = test, output=False)
        plt.plot(yhat[var], linewidth=linewidth)
        print(f'{var} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse, rmse))

    plt.show()


