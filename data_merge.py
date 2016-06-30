### merge flight on-time performance and historical weather data

import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import datetime as dt
import glob
import pickle
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2


# connect to SQL database:
con = None
con = psycopg2.connect(database = dbname, user = username, host='localhost', password=pswd)
engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))

# query: WBAN to Airport mapping file
sql_query = """
    SELECT * FROM wban2airport
    """
wban2airport = pd.read_sql_query(sql_query, con)


## convert local to UTC time at given airports
def local_to_UTC(df, origin_or_dest):
    df_UTC = pd.DataFrame()
    for index in range(df.shape[0]):
        oneRow = df.iloc[index]
              
        if index % 10000 == 0:
            print index,
            
        if origin_or_dest == 'origin':
            hour = str(oneRow['crsdeptime'])[0:2]
        elif origin_or_dest == 'dest':
            hour = str(oneRow['crsarrtime'])[0:2]
        else:
            print "Error: please enter 'origin' or 'dest'"
        
        local_time = dt.datetime(int(oneRow['year']),
                                 int(oneRow['month']),
                                 int(oneRow['dayofmonth']),
                                 int(hour)
                                )
        UTC_time = local_time + dt.timedelta(hours=oneRow['utcoffset_' + origin_or_dest])
        df_UTC = df_UTC.append(pd.Series([int(UTC_time.month), 
                                          int(UTC_time.day), 
                                          int(UTC_time.hour)]
                                        ),
                               ignore_index=True)
        
    df_UTC.columns = [x + '_' + origin_or_dest for x in ['utcmonth', 'utcday', 'utchour']] 
    df_merged = pd.concat([df, df_UTC], axis=1)
    return df_merged


# query: flight on-time performance in 2015 (sample 10%)
sql_query = """
    SELECT year, quarter, month, dayofmonth, dayofweek, flightdate,
        carrier, origin, originairportid, dest, destairportid, distance, 
        crsdeptime, crsarrtime, crselapsedtime, deptimeblk, arrtimeblk, 
        arrdelay, cancelled
    FROM flight_data_2015
    TABLESAMPLE BERNOULLI (10);
    """
flight = pd.read_sql_query(sql_query, con)

# merge in WBAN and UTC times:
chunk = flight
del flight

chunk = pd.merge(chunk, wban2airport, how='inner', left_on='destairportid', right_on='airportid', sort=False)
chunk.columns = list(chunk.columns[0:-2]) + list(chunk.columns[-2:] + '_dest')

chunk = pd.merge(chunk, wban2airport, how='inner', left_on='originairportid', right_on='airportid', sort=False)
chunk.columns = list(chunk.columns[0:-2]) + list(chunk.columns[-2:] + '_origin')

chunk = chunk.ix[pd.notnull(chunk.wban_origin) & pd.notnull(chunk.wban_dest), :]
chunk = chunk.ix[pd.notnull(chunk.year) & pd.notnull(chunk.month) & pd.notnull(chunk.dayofmonth), :]
chunk = chunk.ix[pd.notnull(chunk.crsdeptime) & pd.notnull(chunk.crsarrtime), :]
chunk = chunk.ix[pd.notnull(chunk.arrdelay), :]
chunk = chunk.reset_index()

chunk = local_to_UTC(chunk, 'origin')
chunk = local_to_UTC(chunk, 'dest')

chunk.to_sql('flight_10pct_wban_utc_data_2015', engine, if_exists='replace', index=False)

### merge in weather at the origin airport (SQL commands)
CREATE TABLE flight_10pct_originweather_data_2015 AS
    SELECT * FROM flight_10pct_wban_utc_data_2015 AS f
    INNER JOIN (SELECT wban AS wban_w_origin,
                    month AS month_origin,
                    day AS day_origin,
                    hour AS hour_origin,
                    airtemp AS airtemp_origin, 
                    dewpointtemp AS dewpointtemp_origin, 
                    sealevelpressure AS sealevelpressure_origin,
                    windspeed AS windspeed_origin, 
                    winddirection AS winddirection_origin, 
                    precipdepth1hr AS precipdepth1hr_origin
                FROM weather_data_2015) AS w
    ON (f.wban_origin = w.wban_w_origin 
        AND f.utcmonth_origin = w.month_origin 
        AND f.utcday_origin = w.day_origin
        AND f.utchour_origin = w.hour_origin);

### merge in weather at the destination airport (SQL commands)
CREATE TABLE flight_10pct_weather_data_2015 AS
    SELECT * FROM flight_10pct_originweather_data_2015 AS f
    INNER JOIN (SELECT wban AS wban_w_dest,
                    month AS month_dest,
                    day AS day_dest,
                    hour AS hour_dest,
                    airtemp AS airtemp_dest, 
                    dewpointtemp AS dewpointtemp_dest, 
                    sealevelpressure AS sealevelpressure_dest,
                    windspeed AS windspeed_dest, 
                    winddirection AS winddirection_dest, 
                    precipdepth1hr AS precipdepth1hr_dest
                FROM weather_data_2015) AS w
    ON (f.wban_dest = w.wban_w_dest 
        AND f.utcmonth_dest = w.month_dest 
        AND f.utcday_dest = w.day_dest
        AND f.utchour_dest = w.hour_dest);

