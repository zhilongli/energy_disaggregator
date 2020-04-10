import numpy as np
import pandas as pd
from keras.models import load_model
import h5py


class DAEDisaggregator():

    def __init__(self, model_name, sequence_length):
        '''Load the pre-trained model for the specific appliance

        Parameters
        ----------
        model_name: the .h5 model name (with full file extension) containing the trained weights
        sequence_length : the size of window to use on the aggregate data
        '''
        self.mmax = None
        self.sequence_length = sequence_length
        self.MIN_CHUNK_LENGTH = sequence_length
        self.import_model(model_name)

    def disaggregate(self, mains, output_datastore, **load_kwargs):
        '''Disaggregate mains according to the model learnt.

        Parameters
        ----------
        mains : A pd dataframe of the aggregate power of building
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        meter_metadata : metadata for the produced output
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        # load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        # load_kwargs.setdefault('sample_period', 60)
        # load_kwargs.setdefault('sections', mains.good_sections())

        # for now we also don't need the timeframe, which is the range of the time period 
            # for the specified data. 

        # timeframes = [] 
        # data_is_available = False
        
        # Dont need the chunk stuff for now treat the whole mains dataframe as one chunk
        # for chunk in mains.power_series(**load_kwargs):
        #     if len(chunk) < self.MIN_CHUNK_LENGTH:
        #         continue
        #     print("New sensible chunk: {}".format(len(chunk)))

        #     timeframes.append(chunk.timeframe)
        #     measurement = chunk.name
        #     chunk2 = self._normalize(chunk, self.mmax)

        #     appliance_power = self.disaggregate_chunk(chunk2)
        #     appliance_power[appliance_power < 0] = 0
        #     appliance_power = self._denormalize(appliance_power, self.mmax)

        #     # Append prediction to output
        #     data_is_available = True
        #     cols = pd.MultiIndex.from_tuples([chunk.name])
        #     meter_instance = meter_metadata.instance()
        #     df = pd.DataFrame(
        #         appliance_power.values, index=appliance_power.index,
        #         columns=cols, dtype="float32")
        #     key = '{}/elec/meter{}'.format(building_path, meter_instance)
        #     output_datastore.append(key, df)

        #     # Append aggregate data to output
        #     mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
        #     output_datastore.append(key=mains_data_location, value=mains_df)

        normalized_mains = self._normalize(mains, self.mmax)
        appliance_power = self.disaggregate_chunk(normalized_mains)
        appliance_power[appliance_power<0] = 0
        appliance_power = self._denormalize(appliance_power, self.mmax)
        # now we have the disaggregated data, just append to the mains
        disaggregated_df = pd.concat([mains, appliance_power], axis=1, sort=False)
        return disaggregated_df

        #TODO can combine the disaggregate_chunk function into this since we don't do chunks anymore

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series to disaggregate
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        s = self.sequence_length
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        additional = s - (up_limit % s)
        X_batch = np.append(mains, np.zeros(additional))
        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s ,1))

        pred = self.model.predict(X_batch)
        pred = np.reshape(pred, (up_limit + additional))[:up_limit]
        column = pd.Series(pred, index=mains.index, name="appliance")

        appliance_powers_dict = {}
        appliance_powers_dict["appliance"] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers


    def import_model(self, filename):
        '''Loads keras model from h5

        Parameters
        ----------
        filename : filename for .h5 file

        Returns: Keras model
        '''
        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]


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


    def _pre_disaggregation_checks(self, load_kwargs):
        if not self.model:
            raise RuntimeError(
                "The model needs to be instantiated before"
                " calling `disaggregate`.  For example, the"
                " model can be instantiated by running `train`.")

        if 'resample_seconds' in load_kwargs:
            DeprecationWarning("'resample_seconds' is deprecated."
                               "  Please use 'sample_period' instead.")
            load_kwargs['sample_period'] = load_kwargs.pop('resample_seconds')

        return load_kwargs