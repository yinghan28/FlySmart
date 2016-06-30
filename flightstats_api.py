### query flights based on the most recent schedule

import urllib2
import json
import pandas as pd
import pprint
import datetime as dt


## flights on a given departure or arrival date
def flights_query(departureAirportCode, arrivalAirportCode, Date, DepOrArr):
    year, month, day = Date.split('-')
    
    if DepOrArr == 'Departure':
        direction = 'departing'
    elif DepOrArr == 'Arrival':
        direction = 'arriving'
    else:
        print "Error: please enter 'Departure' or 'Arrival'"
        
    f = urllib2.urlopen('https://api.flightstats.com/flex/schedules/rest/v1/json/from/'
                        + departureAirportCode + '/to/' + arrivalAirportCode
                        + '/' + direction + '/' + year + '/' + month + '/' + day
                        + '?appId=5d2ec7ba&appKey=fc2de9feb8ccb8e929a66bb0fd4657ec')
    json_string = f.read()
    parsed_json = json.loads(json_string)

    flights = pd.DataFrame()
    for flight in parsed_json['scheduledFlights']:
        #pprint.pprint(flight)
        if flight['isCodeshare'] == True:
            continue
        Carrier = flight['carrierFsCode']
        FlightNumber = flight['flightNumber']
        DepDate, DepTime = flight['departureTime'].split('T')
        ArrDate, ArrTime = flight['arrivalTime'].split('T')
        flights = flights.append(pd.Series([Carrier, FlightNumber, 
                                            DepDate, DepTime, 
                                            ArrDate, ArrTime]
                                          ), 
                                 ignore_index=True)
    
    flights.columns = ['Carrier', 'FlightNumber', 'DepDate', 'DepTime', 'ArrDate', 'ArrTime']
    return flights

## flights within 3 days
def flights_3days_query(departureAirportCode, arrivalAirportCode, Date, DepOrArr):
    previousDate = str((dt.datetime.strptime(Date, '%Y-%m-%d') 
                        + dt.timedelta(days = -1)).date())
    nextDate = str((dt.datetime.strptime(Date, '%Y-%m-%d') 
                        + dt.timedelta(days = 1)).date())

    flights_3days = pd.DataFrame()
    for date in [previousDate, Date, nextDate]:
        flights_3days = flights_3days.append(flights_query(departureAirportCode,
                                                           arrivalAirportCode, 
                                                           date, DepOrArr
                                                          ), 
                                             ignore_index=True)
    return flights_3days


if __name__ == '__main__':
    flights_3days_query(departureAirportCode, arrivalAirportCode, Date, DepOrArr)

