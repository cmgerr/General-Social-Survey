import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('../Data/gss_subset_cleaned.csv')

categorical_cols = ['marital', 'sex','divorce', 'dwelown', 'dwelling', 'hhrace']
df.drop(['paeduc', 'maeduc', 'speduc'], axis=1, inplace=True)

# convert data into numpy array
# dn = pd.get_dummies(df_1972, columns=categorical_cols, drop_first=True)
# dn_1972 = dn.as_matrix(columns=None)
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(dn_1972)
#Determine the silhouette coefficient, a metric to test how well each of the data points lies within the cluster
#The best value is 1 and the worst value is -1.
#Values near 0 indicate overlapping clusters.
# Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
#silhouette = metrics.silhouette_score(dn_1972, labels, metric='euclidean')

# removing columns that do not make up more than 90% of a df's years
def one_year_segment_and_clean(df, year):
    df_sub = df[df.year==year]
    df_sub = df_sub.drop('year', axis=1)

    for col in df_sub:
        if df_sub[col].count() < (0.9* len(df_sub[col])):
            del df_sub[col]
    df_sub = df_sub.dropna()

    # capturing new categorical columns after dropping above
    new_cat = []
    for i in categorical_cols:
        for col in df_sub:
            if col == i:
                new_cat.append(col)
    df_sub = pd.get_dummies(df_sub, columns=new_cat, drop_first=True)
    return df_sub

def kmeans_scores_years_individual(df, num_clusters):
    scores = {}
    for yr in np.unique(df.year):
        frame = one_year_segment_and_clean(df, yr)
        matrix = frame.as_matrix(columns=None)
        model = KMeans(n_clusters=num_clusters)
        model.fit(matrix)
        labels = model.labels_
        silhouette = metrics.silhouette_score(matrix, labels, metric='euclidean')
        scores[yr] = silhouette
    return scores

# returning silhouette scores for specified number of clusters(on a year to year basis)
# and returning score for ALL years
kmeans_scores_years_individual(df, 3)

#





# larger grouping of years for silhouette scores, removing columns without a threshold of
# non-null values greater than 90%, dropping rows still containing nan vals, and adding dummy values
# for all categorical columns

def multi_year_segment_and_clean(df, startyear, endyear):
    # defining year segment from inputs
    df_sub = df[(df.year >= startyear) & (df.year <= endyear)]
    df_sub = df_sub.drop('year', axis=1)

    for col in df_sub:
        if df_sub[col].count() < (0.9* len(df_sub[col])):
            del df_sub[col]
    # capturing remaining categorical columns after dropping above
    new_cat = []
    for i in categorical_cols:
        for col in df_sub:
            if col == i:
                new_cat.append(col)

    #dropping remaining nan values
    df_sub = df_sub.dropna()

    #adding dummy columns
    df_sub = pd.get_dummies(df_sub, columns=new_cat, drop_first=True)
    return df_sub


def kmeans_score_year_block(df, num_clusters, startyear, endyear):

    # segementing data & fitting model
    frame = multi_year_segment_and_clean(df, startyear, endyear)
    matrix = frame.as_matrix(columns=None)
    model = KMeans(n_clusters=num_clusters)
    model.fit(matrix)
    labels = model.labels_

    # silhouette score for multi-year block
    silhouette = metrics.silhouette_score(matrix, labels, metric='euclidean')
    return silhouette


# kmeans_score_year_block(df, 3, 1972, 1982)
# returns -> 0.45065992911380309

# same functions as above, but returns label on original segmentation & cleaning
def segment_copy_dummys_with_target(df, startyear, endyear, target):
    # defining year segment from inputs
    df_sub = df[(df.year >= startyear) & (df.year <= endyear)]

    # delete columns below threshold except target
    for col in df_sub:
        if df_sub[col].count() < (0.9* len(df_sub[col])) and col !=target:
            del df_sub[col]

    #dropping remaining nan values
    df_sub = df_sub.dropna()

    # segmenting target series & deleting target from df
    target_series = pd.Series(df_sub[target])
    del df_sub[target]

    # capturing remaining categorical columns after dropping above
    new_cat = []
    for i in categorical_cols:
        for col in df_sub:
            if col == i:
                new_cat.append(col)

    # copy database
    df_og = df_sub.copy()

    #adding dummy columns
    df_sub = pd.get_dummies(df_sub, columns=new_cat, drop_first=True)
    return df_sub, df_og, target_series

# dummies, original, target_vals = segment_copy_dummys_with_target(df, 2010, 2014, 'happy')

def kmeans_labels(df, num_clusters, startyear, endyear, target):

    # segementing data
    dummy_df, og_df, target_values = segment_copy_dummys_with_target(df, startyear, endyear, target)

    # fitting model
    matrix = dummy_df.as_matrix(columns=None)
    model = KMeans(n_clusters=num_clusters)
    model.fit(matrix)
    labels = model.labels_

    # silhouette score for multi-year block
    og_df['labels'] = labels
    og_df[target] = target_values
    return og_df


test = kmeans_labels(df, 3, 2000, 2014, 'happy')

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.grid_search import GridSearchCV

# set random state to use throughout
rs = 25

# import data and format it
data = kmeans_labels(df, 3 , 2005, 2014, 'happy')

# drop highly correlated columns and unusable columns identified in earlier analysis
data.drop(['year', 'hompop', 'earnrs'], axis=1, inplace=True)

# drop less important features from first run-through of RF
data.drop(['babies', 'preteen', 'teens', 'divorce', 'dwelling', 'sex'], axis=1, inplace=True)
data.dropna(inplace=True)
data

################################################################
################################################################
################################################################
# using the clustering labels as a factor
# set X and y
X = pd.get_dummies(data.drop('happy', axis=1), drop_first=True)
y = data['happy'] > 1

# look at % in each class
y.value_counts()/y.count()
y.value_counts()
# do train_test split
X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(),
        stratify = y, test_size=0.2, random_state=rs)

# instantiate Random Forest
rf = RandomForestClassifier(random_state=rs, n_jobs=-1)
data
# define param grid for gridsearch
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [12,15,18],
    'n_estimators': [12,15,18],
    'min_samples_split': [40,50,60],
    'min_samples_leaf': [5,10,20],
    'max_features': [5,10,'auto']}

# instantiate gridsearch for random forest
gsrf = GridSearchCV(rf, param_grid, n_jobs=-1)

gsrf.fit(X_train, y_train)

gsrf.best_estimator_

gsrf_pred = gsrf.predict(X_test)

# predictions with threshold of 50%... baseline of
print classification_report(y_test, gsrf_pred)
print confusion_matrix(y_test, gsrf_pred)


# changing threshold
gsrf_proba = gsrf.predict_proba(X_test)
gsrf_pred2 = gsrf_proba[:,1] > 0.85
print confusion_matrix(y_test, gsrf_pred2)
print classification_report(y_test, gsrf_pred2 )


##############################################################################
##############################################################################
##############################################################################
##############################################################################

# fitting and predicting different models for each cluster

data_1 = data[data['labels']==0]
data_2 = data[data['labels']==1]
data_3 = data[data['labels']==2]


X_1 = pd.get_dummies(data_1.drop('happy', axis=1), drop_first=True)
X_2 = pd.get_dummies(data_2.drop('happy', axis=1), drop_first=True)
X_3 = pd.get_dummies(data_3.drop('happy', axis=1), drop_first=True)

y_1 = data_1['happy'] > 1
y_2 = data_2['happy'] > 1
y_3 = data_3['happy'] > 1

# look at % in each class
y_1.value_counts()/y_1.count()
y_1.value_counts()

y_2.value_counts()/y_2.count()
y_2.value_counts()

y_3.value_counts()/y_3.count()
y_3.value_counts()

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1.as_matrix(), y_1.as_matrix(),
        stratify = y_1, test_size=0.2, random_state=rs)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2.as_matrix(), y_2.as_matrix(),
        stratify = y_2, test_size=0.2, random_state=rs)

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3.as_matrix(), y_3.as_matrix(),
        stratify = y_3, test_size=0.2, random_state=rs)


# instantiate Random Forest
rf1 = RandomForestClassifier(random_state=rs, n_jobs=-1)
rf2 = RandomForestClassifier(random_state=rs, n_jobs=-1)
rf3 = RandomForestClassifier(random_state=rs, n_jobs=-1)

# define param grid for gridsearch
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [12,15,18],
    'n_estimators': [12,15,18],
    'min_samples_split': [40,50,60],
    'min_samples_leaf': [5,10,20],
    'max_features': [5,10,'auto']}

# instantiate gridsearch for random forest
gsrf1 = GridSearchCV(rf1, param_grid, n_jobs=-1)
gsrf2 = GridSearchCV(rf2, param_grid, n_jobs=-1)
gsrf3 = GridSearchCV(rf3, param_grid, n_jobs=-1)

gsrf1.fit(X_train_1, y_train_1)
gsrf1.best_estimator_
gsrf_pred_1 = gsrf1.predict(X_test_1)

# predictions with threshold of 50%... baseline of
print classification_report(y_test_1, gsrf_pred_1)
print confusion_matrix(y_test_1, gsrf_pred_1)

# changing threshold
gsrf_proba_1 = gsrf1.predict_proba(X_test_1)
gsrf2_pred_1 = gsrf_proba_1[:,1] > 0.85

print confusion_matrix(y_test_1, gsrf2_pred_1)
print classification_report(y_test_1, gsrf2_pred_1 )

##############################################################################
# Second Cluster

gsrf2.fit(X_train_2, y_train_2)
gsrf2.best_estimator_
gsrf_pred_2 = gsrf2.predict(X_test_2)

print classification_report(y_test_2, gsrf_pred_2)
print confusion_matrix(y_test_2, gsrf_pred_2)

# changing threshold second cluster
gsrf_proba_2 = gsrf2.predict_proba(X_test_2)
gsrf2_pred_2 = gsrf_proba_2[:,1] > 0.90

print classification_report(y_test_2, gsrf2_pred_2)
print confusion_matrix(y_test_2, gsrf2_pred_2)

##############################################################################
# third cluster
gsrf3.fit(X_train_3, y_train_3)
gsrf3.best_estimator_
gsrf_pred_3 = gsrf3.predict(X_test_3)

print classification_report(y_test_2, gsrf_pred_2)
print confusion_matrix(y_test_2, gsrf_pred_2)

# changing threshold second cluster
gsrf_proba_3 = gsrf3.predict_proba(X_test_3)
gsrf3_pred_2 = gsrf_proba_3[:,1] > 0.85

print classification_report(y_test_3, gsrf3_pred_2)
print confusion_matrix(y_test_3, gsrf3_pred_2)

##############################################################################
##############################################################################
##############################################################################
# predictions with 2 clusters... slightly higher silhouette_score...
# possibly adding more features to the space?
