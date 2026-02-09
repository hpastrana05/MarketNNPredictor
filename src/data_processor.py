import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, data_path, target_column=3, sequence_length=50):
        '''Initializes the DataProcessor with the path to the data file,'''
        self.data_path = data_path
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.data = self.load_data()
        self.dates = self.data['Timestamp']
        
    

    def load_data(self):
        '''Loads data from the CSV file specified by data_path.'''
        data = pd.read_csv(self.data_path)
        return data
    

    def preprocess_data(self, train=0.8, test=None):
        '''
        Preprocesses data by scaling and creating sequences if target_column is specified.
        Assuming that data is cleaned and sorted (Ascending (Oldest data first))
        '''

        if self.target_column:
            # Drop the timestamp as it will be Noise for the NN
            print("Preprocessing data... Dropping Timestamp column.")
            data = self.data.drop(columns=['Timestamp'])
            print("Dropped Timestamp column.")

            # This will assign the idx of train and test
            data_len = len(data)
            if test:
                train_idx = data_len - int(data_len*test)
            else:
                train_idx = int(len(data)*train)

            print(f"Splitting data at index: {train_idx}")
            train_data = data[:train_idx]
            test_data = data[train_idx:]

            # Fit the scaler on training data only
            print("Fitting scaler on training data...")
            self.scaler.fit(train_data)
            print("Scaler fitted.")
            print("Transforming data...")
            scaled_train = self.scaler.transform(train_data).astype(np.float32)
            scaled_test = self.scaler.transform(test_data).astype(np.float32)
            print("Data transformed.")

            # Sequence data creation
            print("Creating sequences...")
            X_train, y_train = self._create_sequences(scaled_train)
            X_test, y_test = self._create_sequences(scaled_test)
            print("Sequences created.")

            self.preprocessed_data = (X_train, y_train, X_test, y_test)

            return X_train, y_train, X_test, y_test
        else:
            print("No target column specified, skipping preprocessing.")
            return
        


    def _create_sequences(self, data):
        '''Creates sequences of data for time series prediction using efficient views.'''
        # Use stride_tricks to avoid creating a massive list and copying memory
        seq_len = self.sequence_length
        
        # Create view: (Batch, Features, Time)
        # sliding_window_view appends the window dimension at the end when axis is specified
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=seq_len, axis=0)
        
        # We drop the last window because its corresponding target (at index N) would be out of bounds
        X = windows[:-1]
        
        # Transpose to get (Batch, Time, Features) from (Batch, Features, Time)
        X = X.transpose(0, 2, 1)
        
        # Targets start from index seq_len
        y = data[seq_len:, self.target_column]
        
        return X, y
    
    def save_scaler(self, name = "scaler.save"):
        '''
        Saves the fitted scaler to a file.
        Try to use the name of the dataset
        '''
        path = f"data/{name}"
        joblib.dump(self.scaler, path)
    
    def save_preprocessed_data(self, name="preprocessed_data.save"):
        '''
        Saves the preprocessed data to a file.
        Try to use the name of the dataset
        '''
        path = f"data/{name}"
        joblib.dump(self.preprocessed_data, path)