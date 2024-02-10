import xgboost as xgb
import polars as pl

class Model:
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
        y_train = self.train_set.select(self.target).to_numpy()

        X_test = self.test_set.select(self.features).to_numpy()
        y_test = self.test_set.select(self.target).to_numpy()

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100
            )
        
    def predict(self):
        return self.model.predict(self.test_set[self.features])

class XGBRegressor(Model):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.model: xgb.XGBRegressor = xgb.XGBRegressor(
            booster='gbtree',
            n_estimators=500,
            early_stopping_rounds=100,
            objective='reg:squarederror',
            max_depth=12,
            learning_rate=0.01
        )