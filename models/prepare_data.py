from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import polars as pl
import numpy as np

class DataSet:
    def __init__(self, data: pl.DataFrame, target_column: str, date_column: str = None, categorical: bool = False, scale: bool = False) -> None:
        self.scaler = MinMaxScaler()
        self.encoder = LabelEncoder()
        self.poly = PolynomialFeatures(degree=2)
        self.date_column = date_column
        self.target_column = target_column
        self._categorical = categorical
        self._scale = scale
        self.X, self.y = self.preprocess_data(data)


    def preprocess_data(self, data: pl.DataFrame):

        if self.date_column:
            x_data = data.drop([self.date_column, self.target_column])
        else:
            x_data = data.drop([self.target_column])  

        y_data = data.get_column(self.target_column)

        if self._categorical:
            # normalize the data
            X = self.scaler.fit_transform(x_data)
            # encode target variable
            y = self.encoder.fit_transform(y_data)
            y = to_categorical(y)  # convert to one-hot encoded format
        
        else:
            if self._scale:
                X = self.scaler.fit_transform(x_data.to_numpy())
                y_reshaped = y_data.to_numpy().reshape(-1, 1)
                y = self.scaler.fit_transform(y_reshaped).flatten()
            else:
                X = x_data.to_numpy()
                y = y_data.to_numpy()


        X = self.poly.fit_transform(X)
        # reshape data for LSTM input [samples, time steps, features]
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])

        return X_reshaped, y

    def split_data(self, test_size=0.2) -> tuple:
        # split data into training and testing sets
        cutoff = int(len(self.X) * (1 - test_size))
        X_train = self.X[:cutoff]
        X_test = self.X[cutoff:]
        y_train = self.y[:cutoff]
        y_test = self.y[cutoff:]
        return X_train, X_test, y_train, y_test
    
    def decode(self, valueArray):
        try:
            return self.encoder.inverse_transform(valueArray)
        except ValueError:
            return self.encoder.inverse_transform(np.argmax(valueArray, axis=1))
    
    def unscale(self, valueArray):
        predictions = valueArray.reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()