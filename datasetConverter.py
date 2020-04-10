import pandas as pd

def convert(filename):
    """Converts the given csv file to a pd dataframe/series, 
        Extraneous columns should be dropped, keeping only timestamp
        and total power used.

        Output: saves to individual pickle files
        
        Assumes the following format of the csv file: 
        - Contains a timestamp column
        - Contains a site_meter column that is the total power usage at the time
        - Each of the appliances occupy a separate column, with all of them summing to site_meter
        - Units for each reading should be W, or otherwise must be consistent"""
    
    print('Loading csv...')
    columns = ['site_meter', 'AHU_Real', 'RTU_Real', 'Indoor Load (W)']
    tz = 'US/Eastern'
    for col in columns:
        df = pd.read_csv(filename, skipinitialspace=True, usecols=["timestamp", col])
        df.index = pd.to_datetime(df["timestamp"])        
        df = df.drop("timestamp", 1)
        df = df.tz_localize(tz, ambiguous='NaT')
        df = df.loc[df.index.notnull()]
        df.to_pickle('{}.pkl'.format(filename+'_'+col))
    