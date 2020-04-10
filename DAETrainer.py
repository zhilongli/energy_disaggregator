from __future__ import print_function, division
from warnings import warn, filterwarnings

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py
import random
import sys

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout
from keras.utils import plot_model

class DAETrainer():
    """Class to build and train a Denoising Autoencoder neural network for energy disaggregation"""

    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.mmax = None
        self.MIN_CHUNK_LENGTH = sequence_length
        self.model = self._create_model(self.sequence_length)

    def train(self, mains, meter, out_name, epochs=1, batch_size=16, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a pd dataframe (or series?) with the aggregate power data for building
        meter : a pd dataframe (or series?) for the meter data
        out_name : the output model filename without .h5 extension
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''
        # main_power_series = mains.power_series(**load_kwargs)
        # meter_power_series = meter.power_series(**load_kwargs)

        # Train chunks
        run = True
        # mainchunk = next(main_power_series)
        # meterchunk = next(meter_power_series)
        if self.mmax == None:
            self.mmax = mains.max().max()

        # while(run):
        mainchunk = self._normalize(mains, self.mmax)
        meterchunk = self._normalize(meter, self.mmax)

        self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
        self.export_model(out_name+'.h5')
        # try:
        #     print("GETTING NEXT CHUNK")
        #     mainchunk = next(main_power_series)
        #     meterchunk = next(meter_power_series)
        # except:
        #     run = False

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        Note that in our applications a chunk is simply all the data
        '''

        s = self.sequence_length
        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk.loc[ix]
        meterchunk = meterchunk.loc[ix]
        # print("main chunk here: ", mainchunk)
        # print("meter chunk here: ", meterchunk)
        mainchunk = mainchunk.loc[~mainchunk.index.duplicated(keep="first")]
        meterchunk = meterchunk.loc[~meterchunk.index.duplicated(keep="first")]

        # Create array of batches
        #additional = s - ((up_limit-down_limit) % s)
        additional = s - (len(mainchunk) % s)
        X_batch = np.append(mainchunk, np.zeros(additional))
        # print("MAIN CHUNK LEN: ", len(mainchunk))
        # print("METER CHUNK LEN: ", len(meterchunk))
        additional = s - (len(meterchunk) % s)
        Y_batch = np.append(meterchunk, np.zeros(additional))

        # print("x batch size is: ", X_batch.shape)
        # print("y batch size is: ", Y_batch.shape)

        # print("X BATCH IS: ")
        # print(X_batch)
        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s, 1))
        # print("final x batch shape is ", X_batch.shape)
        Y_batch = np.reshape(Y_batch, (int(len(Y_batch) / s), s, 1))

        self.model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, shuffle=True)
        print("done fitting")
    
    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])

    def _create_model(self, sequence_len):
        '''Creates the Auto encoder module described in the paper
        '''
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_len, 1), padding="same", strides=1))
        model.add(Flatten())

        # Fully Connected Layers
        model.add(Dropout(0.2))
        model.add(Dense((sequence_len-0)*8, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense((sequence_len-0)*8, activation='relu'))

        model.add(Dropout(0.2))

        # 1D Conv
        model.add(Reshape(((sequence_len-0), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

        model.compile(loss='mse', optimizer='adam')
        plot_model(model, to_file='model.png', show_shapes=True)

        return model

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk
