import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
matplotlib.style.use('ggplot')
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso



# define random state
rs = 98
df = pd.read_csv('../Data/gss_subset_cleaned.csv')
test = pd.read_csv('../Data/gss_subset_cleaned.csv')
test['dwelling'].value_counts()

categorical_cols = ['marital', 'sex','divorce', 'dwelling', 'hhrace', 'dwelown']
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

def get_model_coefs(model, df_year, ycol):
    X = df_year.drop(ycol, axis=1)
    y = df_year[ycol]
    model1 = model
    model1.fit(X, y)
    feat_names = X.columns
    coefs = model.coef_
    #adding model score
    # score =
    result = dict(zip(feat_names, coefs))
    return result

coef_dict = {}
for yr in np.unique(df.year):
    df_year = get_subset(yr, df, 'happy')
    coefs = get_model_coefs(Lasso(alpha=.05), df_year, 'happy')
    coef_dict[yr] = coefs


print coef_dict

coef_df = pd.DataFrame(coef_dict)
coef_df
