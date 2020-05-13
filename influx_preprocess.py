from influxdb import DataFrameClient
import pandas as pd

def mains_getter(building_id, start_date, end_date):
    """Sets up an influxDB connection with client, and get only the mains data
        for the defined time period into a pd dataframe. building_id should 
        be the name of the building as it appears in the database, string format
        Returns the dataframe with one column of the mains power"""

    client = DataFrameClient(host='localhost', port=6086) #our Chronograf is at port 6086
    # then can just make query!
    params = {"start_date":start_date,
            "end_date":end_date,
            "building_id":building_id}

    df = client.query('SELECT mean("power") AS "mean_power", \
                        FROM "Electricity"."rp_sec"."USR_Electricity_SEC" \
                        WHERE time > $start_date AND time < $end_date \
                        AND "deviceID"=$building_id \
                        GROUP BY time(10s) FILL(null)',
                        dropna=True, bind_params=params)
    return df