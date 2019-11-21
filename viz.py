import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

import prepare

df = pd.read_csv('activity.csv')
