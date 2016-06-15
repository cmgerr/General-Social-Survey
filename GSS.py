import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# set a random state to use throughout
rs = 7

# first, read in the descriptions and variable names for all columns
descriptions = pd.read_csv('gss.csv', nrows=1)

# create a dictionary of variable definitions to use as reference
desc_dict = {}
for col in descriptions:
    desc_dict[descriptions[col][0]] = col

# get list of columns
test = pd.read_csv('gss.csv', header=1, nrows=5)
cols = test.columns.tolist()
valid_cols = [x for x in cols if "Unnamed" not in x]
valid_cols

# choose a subset of columns to extract for initial analysis
col_subset = ['year', 'marital', 'sibs', 'childs',
    'age', 'educ', 'paeduc', 'maeduc', 'speduc', 'sex', 'hompop',
    'income', 'earnrs', 'happy']

df = pd.read_csv('gss.csv', header=1, usecols = col_subset)

# look at info
df.info()

# convert marital status and sex to categories, replace
# "didn't answer"/"don't know" type codes for other numeric variables
marital_status = {
    1: 'married',
    2: 'widowed',
    3: 'divorced',
    4: 'separated',
    5: 'never_married',
    9: np.nan
}

replace_dict = {
    'marital': marital_status,
    'sibs': {-1: np.nan, 98: np.nan, 99: np.nan},
    'childs': {9: np.nan},
    'educ': {98: np.nan, 99: np.nan},
    'paeduc': {98: np.nan, 99: np.nan},
    'maeduc': {98: np.nan, 99: np.nan},
    'speduc': {98: np.nan, 99: np.nan},
    'sex': {1: 'Male', 2: 'Female'},
    'hompop': {99: np.nan},
    'income': {13: np.nan, 0: np.nan, 98: np.nan, 99: np.nan},
    'earnrs': {9: np.nan},
    'happy': {0: np.nan, 8: np.nan, 9: np.nan}
}

df.replace(to_replace = replace_dict, inplace=True)

# look at info again
df.info()

# what happens if we just ignore nulls for now?
nonnull = df.dropna()
nonnull.info()

# create dummies for 'marital' and 'sex'

nonnull = pd.get_dummies(nonnull, columns =['marital', 'sex'], drop_first=True)

# play around with predicting income
target = nonnull.income
features = nonnull.drop(['year','income'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features,
    target, test_size=0.2, random_state=rs)

# linear model
lm = LinearRegression(n_jobs=-1)
lm.fit(X_train, y_train)
lm.score(X_train, y_train)
lm.score(X_test, y_test)
# not so great!

rf = RandomForestRegressor(max_depth = 8, random_state = rs, verbose = 1, n_jobs=-1)
rf.fit(X_train, y_train)

rf.score(X_train, y_train)
rf.score(X_test, y_test)
feat_imp = pd.DataFrame({'feature': features.columns, 'importance': rf.feature_importances_})
feat_imp.sort_values(by = 'importance', axis = 0, ascending = False)
