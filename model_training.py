### feature engineering and model training

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *
from sklearn import (preprocessing, cross_validation, metrics, linear_model, ensemble)
from statsmodels.distributions.empirical_distribution import ECDF
import datetime as dt
import glob
import pickle
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2


os.chdir('/Users/Serena/Career/Insight/FlySmart')


# connect to SQL database:
with open('local_db_credentials', 'r') as f:
    credentials = f.readlines()
        f.close()

dbname = 'flight_db'
user = credentials[0].rstrip()
pswd = credentials[1].rstrip()

con = None
con = psycopg2.connect(database = dbname, user = user, host='localhost', password=pswd)


# query: flight and weather data in 2015 (sample 50%)
sql_query = """
    SELECT * FROM flight_10pct_weather_data_2015
    TABLESAMPLE BERNOULLI (50);
    """
data = pd.read_sql_query(sql_query, con)


# query: US holidays in 2015
sql_query = """
    SELECT * FROM holidays
    WHERE date LIKE '2015%';
    """
holidays = pd.read_sql_query(sql_query, con)


###########################
### feature engineering ###
###########################

# create dummy variables for categorical features
def create_dummy_var(df, var):
    dummy_df = pd.get_dummies(df[var])
    dummy_df.columns = [var + '_' + str(x) for x in dummy_df.columns]
    return dummy_df.drop(dummy_df.columns[0], axis=1)


# calculate days from nearest holiday
holidays_datetime = [dt.datetime.strptime(x, '%Y-%m-%d') for x in holidays['date']]
daysfromholiday = []
for index in data.index:
    flight_datetime = dt.datetime.strptime(data.iloc[index]['flightdate'], '%Y-%m-%d')
    days =  min([abs(flight_datetime - x) for x in holidays_datetime]).days
    daysfromholiday.append(days)
data['daysfromholiday'] = daysfromholiday

# calculate flight speed
data['speed'] = data['distance'].astype(float).divide(data['crselapsedtime'].astype(float))

# recode missing values
data = data.replace(-9999, np.nan)

# features to include in the model
weather_features = ['airtemp', 'dewpointtemp', 'sealevelpressure', 'winddirection', 'windspeed', 'precipdepth1hr']

continuous_features = ['crselapsedtime', 'distance', 'daysfromholiday', 'speed']     + [x + '_origin' for x in weather_features]     + [x + '_dest' for x in weather_features]

categorical_features = ['carrier', 'origin', 'dest', 
                        'quarter', 'month', 'dayofmonth','dayofweek',  
                        'deptimeblk', 'arrtimeblk']
target = ['arrdelay']

# select features from the original data
data = data[continuous_features + categorical_features + target]

# drop samples with missing values
data = data.dropna(axis=0, how='any')

# convert int64 to int
features_to_int = data[categorical_features].describe().columns.tolist()
for feature in features_to_int:
    data.loc[:, feature] = data.loc[:, feature].astype(int)

# create dummy variables for categorical features
d_categorical = pd.DataFrame()
for feature in categorical_features:
    d_categorical = pd.concat([d_categorical, create_dummy_var(data, feature)], axis=1)

data = pd.concat([data[continuous_features], d_categorical, data[target]], axis=1)

# save all the features used in model training
with open("features.pkl", "wb") as f:
    pickle.dump(list(data.columns), f)
    f.close()

######################
### model building ###
######################

# train/test split
np.random.seed(2016)
splitter = np.random.choice([0, 1, 2], data.size, p=[0.8, 0.1, 0.1])
s = pd.Series(pd.Categorical.from_codes(splitter, categories=["train", "valid", "test"]))

data_train = data.loc[s == 'train']
data_valid = data.loc[s == 'valid']
data_test = data.loc[s == 'test']
del data

# save validation and test datasets
with open("ValidationData.pkl", "wb") as f:
    pickle.dump(data_valid, f)
    f.close()
with open("TestData.pkl", "wb") as f:
    pickle.dump(data_test, f)
    f.close()

# features and target
X_train = data_train.drop('arrdelay', axis=1)
Y_train = (data_train['arrdelay'] > 30).astype(int)

X_valid = data_valid.drop('arrdelay', axis=1)
Y_valid = (data_valid['arrdelay'] > 30).astype(int)

X_test = data_test.drop('arrdelay', axis=1)
Y_test = (data_test['arrdelay'] > 30).astype(int)

# downsample majority class
data_train_downsampled = pd.concat([data_train[Y_train==0].sample(sum(Y_train==1)),
                                    data_train[Y_train==1]], 
                                   axis=0)
X_train_downsampled = data_train_downsampled.drop('arrdelay', axis=1)
Y_train_downsampled = (data_train_downsampled['arrdelay'] > 30).astype(int)

# parameter tuning
#for n_estimators in [10, 50, 100, 150, 200]:
n_estimators = 100
#for max_depth in [20, 30, 40, 50, 60]:
max_depth = 40
for min_samples_split in [10, 30, 50, 70, 90]:
#min_samples_split = 30
    rfc = ensemble.RandomForestClassifier(n_estimators=n_estimators, 
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          oob_score=False,
                                          n_jobs=-1, 
                                          class_weight='balanced')
    kf = cross_validation.KFold(X_train_downsampled.shape[0], n_folds=5, shuffle=True)
    score = cross_validation.cross_val_score(rfc, X_train_downsampled, Y_train_downsampled, 
                                             cv=kf, n_jobs=-1, scoring='roc_auc')
    print min_samples_split, score.mean()

# model fitting on the entire training set
rfc = ensemble.RandomForestClassifier(n_estimators=100,
                                      max_depth=40,
                                      min_samples_split=30,
                                      oob_score=False,
                                      n_jobs=1,
                                      class_weight='balanced')
rfc.fit(X_train_downsampled, Y_train_downsampled)

with open("RandomForest.pkl", "wb") as f:
    pickle.dump(rfc, f)
    f.close()


# performance metric on the test data
Y_test_predicted_proba = rfc.predict_proba(X_test)[:,1]
print metrics.roc_auc_score(Y_test, Y_test_predicted_proba)

# normalize predicted probabilities on the validation data
Y_valid_predicted_proba = rfc.predict_proba(X_valid)[:,1]
ecdf = ECDF(Y_valid_predicted_proba)

with open("ECDF_ValidationData.pkl", "wb") as f:
    pickle.dump(ecdf, f)

# feature importance
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
df_importance = pd.DataFrame([X_test.columns, importances, std]).transpose()

f = open('/Users/Serena/Career/Insight/FlySmart/RandomForest_FeatureImportance.csv', 'w')
df_importance.to_csv(f)
