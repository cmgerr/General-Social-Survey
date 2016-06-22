# normal imports
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

# ANN imports
from keras.models import Sequential
from keras.layers import Dense, Activation
# set random state to use
rs = 25
# import data and format it
data = pd.read_csv('../Data/gss_subset_cleaned.csv')
data = data[data['year']> 2005]
data.drop(['paeduc', 'maeduc', 'speduc', 'income', 'satjob', 'goodlife','health', 'year'], axis=1, inplace=True)
data = pd.get_dummies(data, drop_first=True)
data.info()
# for initial model, just drop all na
data.dropna(inplace=True)
data.shape


# set X, y
target = 'dwelown_owns'
y = data[target]
X = data.drop(target, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(),
    test_size=0.2, random_state=rs)
X_train.shape


# create model
model = Sequential()
model.add(Dense(1, input_dim=27))
model.add(Activation('sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# train model
model.fit(X_train, y_train, nb_epoch = 25, batch_size = 100)

model.evaluate(X_test, y_test)
