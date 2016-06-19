import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
matplotlib.style.use('ggplot')
from sklearn.linear_model import LogisticRegression, LinearRegression

# define random state
rs = 98
df = pd.read_csv('../Data/gss_subset_cleaned.csv')

categorical_cols = ['marital', 'divorce', 'sex']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df.drop(['paeduc', 'maeduc', 'speduc'], axis=1, inplace=True)
df.info()

sns.heatmap(df.corr())

def get_subset(year, df_whole, ycol):
    result = df_whole[df_whole.year==year]
    result = result.drop('year', axis=1)
    for col in result:
        if result[col].count() < (0.9* len(result[col])):
            if col != ycol:
                del result[col]
    result = result.dropna()
    return result

def get_linear_model_coefs(df_year, ycol):
    X = df_year.drop(ycol, axis=1)
    y = df_year[ycol]
    model = LinearRegression(normalize=True, n_jobs=-1)
    model.fit(X, y)
    feat_names = X.columns
    coefs = model.coef_
    result = dict(zip(feat_names, coefs))
    return result

coef_dict = {}
for yr in np.unique(df.year):
    df_year = get_subset(yr, df, 'happy')
    coefs = get_linear_model_coefs(df_year, 'happy')
    coef_dict[yr] = coefs

print coef_dict
