import pandas as pd 

from os import path

import warnings
warnings.filterwarnings('ignore')

#df = pd.read_csv('activity.csv')

def prep_df(df, use_cache = True):
    if use_cache and path.exists('prep_activity.csv'):
        df = pd.read_csv('prep_activity.csv')
        #fmt = '%m/%d/%y'
        df.Date = pd.to_datetime(df.Date)
        df.set_index('Date', inplace = True)
        return df
    
    fmt = '%m/%d/%y'
    df.Date = pd.to_datetime(df.Date, format = fmt)

    for var in df.select_dtypes('object'):
        for i in df[var]:
            df[var][df[var] == i] = i.replace(',',"")
        df[var] = df[var].astype('int')

    df = df.sort_values(by='Date')
    df.set_index('Date', inplace = True)
    df.to_csv('prep_activity.csv', index = True)
    return df

#df = prep_df(df)