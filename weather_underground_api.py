
# coding: utf-8

# In[1]:

import urllib2
import json
import pandas as pd
import pprint


# In[19]:

# Year = 2016
# Month = 6
# Day = 16
# Hour = 11
# Airport = 'ORD'


# In[20]:

def get_10days_weather(Airport):
    if Airport == 'ORD':
        Airport = 'CHI'
        
    f = urllib2.urlopen('http://api.wunderground.com/api/ea777ce9179233ac/hourly10day/q/' + Airport + '.json')
    json_string = f.read()
    parsed_json = json.loads(json_string)
    f.close()

    return parsed_json


# In[4]:

def get_forecast(parsed_json, Year, Month, Day, Hour):    
    for index in range(len(parsed_json['hourly_forecast'])):
        localtime = parsed_json['hourly_forecast'][index]['FCTTIME']
        #print int(weather['mon']), int(weather['mday']), int(weather['hour'])
        if (int(localtime['mon']) == Month) & (int(localtime['mday']) == Day) & (int(localtime['hour']) == Hour):
            break

    weather = parsed_json['hourly_forecast'][index]
    return weather


# In[5]:

def parse_weather(weather):
    parsed_weather = pd.Series([int(weather['temp']['metric']) * 10, 
                              int(weather['dewpoint']['metric']) * 10,
                              int(weather['mslp']['metric']) * 10,
                              weather['wdir']['degrees'],
                              round(int(weather['wspd']['metric']) * 2.78),
                              int(weather['qpf']['metric']) * 10
                             ], 
                             index=['AirTemp', 
                                    'DewPointTemp', 
                                    'SeaLevelPressure', 
                                    'WindDirection', 
                                    'WindSpeed', 
                                    'PrecipDepth1hr'
                                   ])

    return parsed_weather


# In[21]:

if __name__ == '__main__':
    parsed_json = get_10days_weather(Airport)
    weather = get_forecast(parsed_json, Year, Month, Day, Hour)
    weather = parse_weather(weather)


# In[ ]:



