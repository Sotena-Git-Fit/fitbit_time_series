import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

import prepare



# df = pd.read_csv('activity.csv')

# df = prepare.prep_df(df)

def last_max_date(df,var):
    return df[var][df[var] == df[var].max()].index[-1]

# last_max_date(df,'Minutes Sedentary')

def plot_figure_vlines(df,var1, var2):
    plt.figure(figsize = (16,4))
    plt.plot(df[var1])
    plt.plot(df[var2])
    plt.xlabel('Date')
    plt.ylabel('Minutes')
    plt.title('Minutes Very Active & Sedentary')
    plt.vlines(x = last_max_date(df, var2), ymin = min(df[var1]), ymax = max(df[var2]))
    plt.vlines(x = '2018-11-01', ymin = min(df[var1]), ymax = max(df[var2]))
    plt.vlines(x = '2018-11-12', ymin = min(df[var1]), ymax = max(df[var2]))
    plt.vlines(x = '2018-05-31', ymin = min(df[var1]), ymax = max(df[var2]))

# plot_figure_vlines(df, 'Minutes Very Active', 'Minutes Sedentary')

def plot_figure(df,var1, var2):
    plt.figure(figsize = (16,4))
    plt.plot(df[var1])
    plt.plot(df[var2])
    plt.xlabel('Date')
    plt.ylabel('{}/{}'.format(var1,var2))

#plot_figure(df,'Calories Burned', 'Minutes Sedentary')

# plot_figure(df,'Calories Burned', 'Activity Calories')

# high_corr = df[['Calories Burned', 'Steps', 'Minutes Very Active', 'Activity Calories']].corr()

def plot_corr(df,high_corr):
    for x in high_corr:
        for y in high_corr:
            plt.scatter(x = df[x], y = df[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.show()

#plot_corr(df,high_corr)

def plot_hist(df):
    for col in df.columns:
        df[col].hist()
        plt.title('Distribution of {}'.format(col))
        plt.ylabel('Count')
        plt.xlabel('{}'.format(col))
        plt.show()

# plot_hist(df)

def plot_columns(df):
    for col in df.columns:
        plt.figure(figsize = (16,4))
        plt.plot(df.resample('W').sum()[col])
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.show()

# plot_columns(df)
