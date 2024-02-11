import numpy as np
import polars as pl
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader

import keras
from keras import layers

class XGBModel:
    def __init__(self,
                 train_set: pl.DataFrame,
                 test_set: pl.DataFrame,
                 features: list[str],
                 target: list[str]
                 ) -> None:
        
        self.model: xgb.XGBModel
        self.train_set = train_set
        self.test_set = test_set
        self.features = features
        self.target = target
    
    def train(self):

        X_train = self.train_set.select(self.features).to_numpy()
        X_test = self.test_set.select(self.features).to_numpy()

        if isinstance(self.model, xgb.XGBRegressor):
            y_test = self.test_set.select(self.target).to_numpy()
            y_train = self.train_set.select(self.target).to_numpy()

        elif isinstance(self.model, xgb.XGBClassifier):
            labels = {
                'buy': 0,
                'hold': 1,
                'sell': 2
            }

            y_train = self.train_set.get_column(self.target[0]).to_list()
            y_train = [labels[item] for item in y_train]
            y_test = self.test_set.get_column(self.target[0]).to_list()
            y_test = [labels[item] for item in y_test]
            
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100
            )
        
        return {0: 'buy', 1: 'hold', 2: 'sell'}
        
    def predict(self):
        return self.model.predict(self.test_set[self.features])
    
    def feature_importances(self):
        data = {k: v for k, v in zip(self.features, self.model.feature_importances_)}
        return data


class XGBRegressor(XGBModel):

    def __init__(self, params: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.model: xgb.XGBRegressor = xgb.XGBRegressor(**params)

class XGBClassifier(XGBModel):

    def __init__(self, params: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.model: xgb.XGBClassifier = xgb.XGBClassifier(**params)

class LSTM:

    def __init__(self) -> None:
        self.model: keras.Sequential

    def train(self, X_train, y_train, X_test, y_test):

        history = self.model.fit(
            X_train,
            y_train,
            batch_size=64,
            epochs=100,
            validation_data=(X_test, y_test)
            )
        
class CategoricalLSTM(LSTM):

    def __init__(
            self,
            dim: tuple[int],
            n_classes: int,
            params: dict = {},
            **kwargs
            ) -> None:
        
        super().__init__()
        self.model = self.initialize_model(dim, n_classes)

    def initialize_model(self,  dim: tuple[int], n_classes: int):

        model = keras.Sequential()
        model.add(layers.LSTM(64, input_shape=dim))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(n_classes, activation='softmax'))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )
        return model
    
    def predict(self, test_set):
        return np.argmax(self.model.predict(test_set), axis=1)
    

class RegressiveLSTM(LSTM):
    def __init__(
            self,
            dim: tuple[int],
            params: dict = {},
            **kwargs
            ) -> None:
        
        super().__init__()
        self.model = self.initialize_model(dim)
    
    def initialize_model(self,  dim: tuple[int]):
        # Build the LSTM model
        model = keras.Sequential()
        model.add(layers.LSTM(50, return_sequences=True, input_shape=dim))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(50, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(50))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
        
    def predict(self, test_set):
        return self.model.predict(test_set)
    

    
