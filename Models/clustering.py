import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt


df = pd.read_csv('../Data/gss_subset_cleaned.csv')
df

categorical_cols = ['marital', 'sex','divorce', 'dwelown', 'dwelling', 'hhrace']
df.drop(['paeduc', 'maeduc', 'speduc'], axis=1, inplace=True)

from sklearn.cluster import KMeans

# convert data into numpy array
# dn = pd.get_dummies(df_1972, columns=categorical_cols, drop_first=True)
# dn_1972 = dn.as_matrix(columns=None)
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(dn_1972)
#Determine the silhouette coefficient, a metric to test how well each of the data points lies within the cluster
#The best value is 1 and the worst value is -1.
#Values near 0 indicate overlapping clusters.
# Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
silhouette = metrics.silhouette_score(dn_1972, labels, metric='euclidean')

# removing columns that do not make up more than 90% of a df's years
def remove_bad_segment_year(df, year):
    df_sub = df[df.year==year]
    df_sub = df_sub.drop('year', axis=1)
    for col in df_sub:
        if df_sub[col].count() < (0.9* len(df_sub[col])):
            del df_sub[col]
    df_sub = df_sub.dropna()
    new_cat = []
    for i in categorical_cols:
        for col in df_sub:
            if col == i:
                new_cat.append(col)
    df_sub = pd.get_dummies(df_sub, columns=new_cat, drop_first=True)
    return df_sub


def kmeans_scores_years(df, num_clusters):
    scores = {}
    for yr in np.unique(df.year):
        frame = remove_bad_segment_year(df, yr)
        matrix = frame.as_matrix(columns=None)
        model = KMeans(n_clusters=num_clusters)
        model.fit(matrix)
        labels = model.labels_
        silhouette = metrics.silhouette_score(matrix, labels, metric='euclidean')
        scores[yr] = silhouette
    return scores

kmeans_scores_years(df, 2)

years =[]
for i in np.unique(df.year):
    years.append(i)
