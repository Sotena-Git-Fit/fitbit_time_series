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

import pandas as pd
import numpy as np

import math

def train_test_split(df):
    last_dy = df[-1:].index[0]
    two_wks_bf_last_dy = (last_dy - timedelta(14))
    train = df[:two_wks_bf_last_dy]    
    test = df[two_wks_bf_last_dy + timedelta(1):]    
    return train, test

def plot_train_test(train, test, df):
    plt.figure(figsize = (16,4))
    plt.plot(train[df.columns], color = 'black')
    plt.plot(test[df.columns])
    plt.show()





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

    for var in target_vars:
        plt.rc('font', size = 14)
        plt.figure(figsize=(16, 8))
        plt.plot(train[var],label='Train - {}'.format(var), linewidth=1)
        plt.plot(test[var], label='Test - {}'.format(var), linewidth=1)
        plt.plot(yhat[var], linewidth=linewidth)
        plt.xlabel('Date')
        plt.ylabel(var)
        plt.title('Predict {}'.format(var))
        plt.show()
        mse, rmse = evaluate(yhat, var, train, test, output=False)
        print(f'{var} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse, rmse))


eval_df = pd.DataFrame(columns = ['model_type','target_var','metric', 'value'])

def append_eval_df(yhat, model_type, train, test):
    temp_eval_df = pd.concat([pd.DataFrame([[model_type,i,'mse', evaluate(yhat, target_var = i, train=train, test = test, output = False)[0]],
                                            [model_type,i,'rmse', evaluate(yhat, target_var= i, train=train, test = test, output = False)[1]]],
                                            columns = ['model_type', 'target_var', 'metric', 'value'])
                             for i in train.columns], ignore_index=True)
    return eval_df.append(temp_eval_df,ignore_index=True)
    

def last_observed_yhat(yhat,train):
    for var in train.columns:
        yhat[var] = float(train[var][-1:])
    return yhat

def simple_average(yhat, train):
    for i in train.columns:
        yhat[i] = train[i].mean()
    return yhat

def moving_average(yhat, train, periods):
    for i in train.columns:
        yhat[i] = train[i].rolling(periods).mean()[-1]
    return yhat

def holt_linear_trend(yhat, train, test):
    for var in train.columns:
        models = Holt(train[var]).fit(smoothing_level=.3, smoothing_slope=.1, optimized=False)
        yhat[var] = pd.DataFrame(models.forecast(test[var].shape[0]), columns=[var])
    return yhat


