
# coding: utf-8

# In[29]:

### prediction on user input data using random forest

import os
import numpy as np
import pandas as pd
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
import datetime as dt
import pickle
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
from weather_underground_api import *
from flightstats_api import flights_3days_query


# In[2]:

# features = pickle.load(open("/Users/Serena/Career/Insight/FlySmart/features.pkl", "rb"))
# rfc = pickle.load(open("/Users/Serena/Career/Insight/FlySmart/RandomForest.pkl", "rb"))
# ecdf = pickle.load(open("/Users/Serena/Career/Insight/FlySmart/ECDF_ValidationData.pkl", "rb"))


# In[3]:

# with open('local_db_credentials', 'r') as f:
#    credentials = f.readlines()
#    f.close()
    
# dbname = 'flight_db'
# user = credentials[0].rstrip()
# pswd = credentials[1].rstrip()

# con = None
# con = psycopg2.connect(database = dbname, user = username, host='localhost', password=pswd)


# In[ ]:

features = pickle.load(open("/home/ubuntu/features.pkl", "rb"))
rfc = pickle.load(open("/home/ubuntu/RandomForest.pkl", "rb"))
ecdf = pickle.load(open("/home/ubuntu/ECDF_ValidationData.pkl", "rb"))


# In[ ]:

def connect_to_db():
    
    host   = 'postgresinstance.c0dzm7nvb0nh.us-west-2.rds.amazonaws.com:5432'
    dbname = 'flysmart'
    
    with open('/home/ubuntu/db_credentials', 'r') as f:
        credentials = f.readlines()
        f.close()
        
    user = credentials[0].rstrip()
    pswd = credentials[1].rstrip()
    
    connection = psycopg2.connect(
        database=dbname,
        user=user,
        password=pswd,
        host=host.split(':')[0],
        port=5432)
    
    return connection


# In[4]:

# connect:
con = connect_to_db()


# In[64]:

# User input
Origin = 'LAX'
Dest = 'SFO'
Date = '2016-06-21'
Time = '12'
DepOrArr = 'Arrival'


# In[6]:

weather_features = ['airtemp', 'dewpointtemp', 'sealevelpressure', 'winddirection', 
                    'windspeed', 'precipdepth1hr']
continuous_features = ['crselapsedtime', 'distance', 'daysfromholiday', 'speed']                     + [x + '_origin' for x in weather_features]                     + [x + '_dest' for x in weather_features]
categorical_features = ['quarter', 'month', 'dayofmonth', 'dayofweek', 
                        'deptimeblk', 'arrtimeblk', 
                        'origin', 'dest', 'carrier']


# In[7]:

def calculate_daysfromholiday(flight_datetime):
    # query: US holidays in 2016
    sql_query = """
        SELECT * FROM holidays
        WHERE date LIKE '2016%';
        """
    holidays = pd.read_sql_query(sql_query, con)
    holidays_datetime = [dt.datetime.strptime(x, '%Y-%m-%d') for x in holidays['date']]
    days_from_holiday = min([abs(flight_datetime - x) for x in holidays_datetime]).days
    return days_from_holiday


# In[8]:

def distance_between_airports(Origin, Dest):
    # query: distance between airports
    sql_query = """
        SELECT distance FROM distance_between_airports
        WHERE origin = '""" + Origin + """' AND dest = '""" + Dest + """';"""
    distance = pd.read_sql_query(sql_query, con)
    return distance.ix[0, 0]


# In[9]:

def createTimeBulk(Hour):
    if Hour >= 0 and Hour <= 5:
        TimeBulk = '0001-0559'
    else:
        Hour_padded = '0'*(2 - len(str(Hour))) + str(Hour)
        TimeBulk = Hour_padded + '00-' + Hour_padded + '59'
    return TimeBulk


# In[10]:

def create_dummy_var(df, desired_features, categorical_feature):
    dummy_df = pd.DataFrame()
    columns = []
    for feature in desired_features:
        if feature.startswith(categorical_feature):
            #print feature.split('_')[1]
            columns.append(feature)
            dummy_column = (df[categorical_feature] == feature.split('_')[1]).astype(int)
            dummy_df = pd.concat([dummy_df, dummy_column], axis=1)
    dummy_df.columns = columns
    return dummy_df


# In[191]:

def delay_prob_prediction(Origin, Dest, Date, Time, DepOrArr):
    
    input_datetime = dt.datetime.strptime(Date + ' ' + Time, '%Y-%m-%d %H')
    
    flights_3days = flights_3days_query(Origin, Dest, Date, DepOrArr)
    Origin_weather_10days = get_10days_weather(Origin)
    Dest_weather_10days = get_10days_weather(Dest)
    
    flight_schedule = pd.DataFrame()
    data = pd.DataFrame()
    for i in range(flights_3days.shape[0]):
        oneRow = flights_3days.loc[i]
        
        Dep_datetime = dt.datetime.strptime(oneRow['DepDate'] + ' ' + oneRow['DepTime'], 
                                            '%Y-%m-%d %H:%M:%S.%f'
                                           )
        Arr_datetime = dt.datetime.strptime(oneRow['ArrDate'] + ' ' + oneRow['ArrTime'], 
                                            '%Y-%m-%d %H:%M:%S.%f'
                                           )
        
        # drop flights outside of +/- 12hrs from user's input
        if DepOrArr == 'Departure':
            if Dep_datetime < input_datetime + dt.timedelta(hours=-12):
                continue
            elif Dep_datetime > input_datetime + dt.timedelta(hours=12):
                continue       
            else:
                pass
        elif DepOrArr == 'Arrival':
            if Arr_datetime < input_datetime + dt.timedelta(hours=-12):
                continue
            elif Arr_datetime > input_datetime + dt.timedelta(hours=12):
                continue
            else:
                pass
        else:
            print "Error: please enter 'Departure' or 'Arrival'"
            
        # air time
        CRSElapsedTime = int((Arr_datetime - Dep_datetime).total_seconds() / 60)
        
        # departure date
        DepYear, DepMonth, DepDay = oneRow['DepDate'].split('-')
        DayofMonth = str(int(DepDay))
        DayOfWeek = str(Dep_datetime.weekday() + 1)
            
        # arrival date
        ArrYear, ArrMonth, ArrDay = oneRow['ArrDate'].split('-')
        
        # departure / arrival time
        DepHour = int(oneRow['DepTime'].split(':')[0])
        ArrHour = int(oneRow['ArrTime'].split(':')[0])
        DepTimeBlk = createTimeBulk(DepHour)
        ArrTimeBlk = createTimeBulk(ArrHour)
        
        # airport weather
        Origin_weather = parse_weather(get_forecast(Origin_weather_10days, int(DepYear), int(DepMonth), int(DepDay), DepHour))
        Origin_weather.index = Origin_weather.index + '_Origin'
        Dest_weather = parse_weather(get_forecast(Dest_weather_10days, int(ArrYear), int(ArrMonth), int(ArrDay), ArrHour))
        Dest_weather.index = Dest_weather.index + '_Dest'
        
        # flight info
        flight = pd.Series({
                    'CRSElapsedTime' : int(CRSElapsedTime), 
                    'DayofMonth' : DayofMonth, 
                    'DayOfWeek' : DayOfWeek,
                    'DepTimeBlk' : DepTimeBlk,
                    'ArrTimeBlk' : ArrTimeBlk,
                    'Carrier' : oneRow['Carrier']
                    })
        
        ## input data
        data_oneRow = pd.concat([flight, Origin_weather, Dest_weather], axis=0)
        data = data.append(data_oneRow, ignore_index=True)
        
        ## flight schedule
        schedule = pd.Series({
                'Carrier' : oneRow['Carrier'],
                'FlightNumber' : oneRow['FlightNumber'], 
                'Origin' : Origin,
                'Dest' : Dest,
                'Dep_datetime': str(Dep_datetime)[:-3],
                'Arr_datetime': str(Arr_datetime)[:-3]
            })
        flight_schedule = flight_schedule.append(schedule, ignore_index=True)
    
    # assign month to quarter
    Month = input_datetime.month
    if Month in [1,2,3]:
        Quarter = 1
    elif Month in [4,5,6]:
        Quarter = 2
    elif Month in [7,8,9]:
        Quarter = 3
    else:
        Quarter = 4
    
    data['Month'] = str(Month)
    data['Quarter'] = str(Quarter)
    data['Origin'] = Origin
    data['Dest'] = Dest
    data['Distance'] = int(distance_between_airports(Origin, Dest))
    data['Speed'] = data['Distance'].divide(data['CRSElapsedTime'])
    data['DaysFromHoliday'] = calculate_daysfromholiday(input_datetime)
    data.columns = data.columns.str.lower()
    
    # create dummy variables for categorical features
    d = pd.DataFrame()
    for categorical_feature in categorical_features:
        d = pd.concat([d, create_dummy_var(data, features, categorical_feature)], axis=1)  
    d = pd.concat([data.drop(categorical_features, axis=1), d], axis=1)
    
    ## predictive probability (target: delay > 30min)
    p = ecdf(rfc.predict_proba(d)[:, 1])
    #p = rfc.predict_proba(d)[:, 1]
    #pct = [str(int(round(x*100))) + '%' for x in p]
    flight_schedule['ProbDelay'] = p
    return flight_schedule


# In[192]:

if __name__ == '__main__':
    #data = delay_prob_prediction(Origin, Dest, Date, Time, DepOrArr)

    #str(int(round(p*100))) + '%'
    p = delay_prob_prediction(Origin, Dest, Date, Time, DepOrArr)
    #print 'This flight has a %.0f%% chance of delay.' % (100 * p)




