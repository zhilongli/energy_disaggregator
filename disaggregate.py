import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from DAEDisaggregator import DAEDisaggregator

if __name__ == "__main__":
    win_size = 200
    disagg = DAEDisaggregator('trained_AHU.h5', win_size)
    #Load pickle files
    mains = pd.read_pickle('data/site_meter.pkl')
    AHU = pd.read_pickle('data/AHU_Real.pkl')
    RTU = pd.read_pickle('data/RTU_Real.pkl')
    indoor_load = pd.read_pickle('data/Indoor Load (W).pkl')
    # set training and testing data
    train_mains = mains.loc[ :"2018-11-15"]
    test_mains = mains.loc["2018-11-15": ]
    train_AHU = AHU.loc[ :"2018-11-15"]
    test_AHU = AHU.loc["2018-11-15": ]
    result_df = disagg.disaggregate(test_mains, "")
    from matplotlib import pyplot as plt

    pred = result_df['appliance'].to_numpy()
    actual = test_AHU['AHU_Real'].to_numpy()
    actual_total = test_mains['site_meter'].to_numpy()
    plt.plot(pred, label='predicted')
    plt.plot(actual_total, label="total usage")
    plt.plot(actual, label='true usage')
    plt.legend(loc='upper right')
    plt.show()
