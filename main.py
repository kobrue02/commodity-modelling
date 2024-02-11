import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yaml

from dateutil.relativedelta import relativedelta
from pathlib import Path

from models.models import *
from models.prepare_data import DataSet
from data.dataset import MergedDataset
from data.util.exceptions.exceptions import *
from data.gold.gold_price_data import GoldPriceData
from data.oil.oil_price_data import OilPriceData
from data.sp500.sp_500_price_data import SP500Data
from data.news.news_sentiment import NewsSentiment


def xgb(train, test):
    params = yaml.safe_load(Path('models/model_config.yml').read_text())

    model = XGBClassifier(
        params=params,
        train_set=train,
        test_set=test,
        features=[
            'sp500_price',
            'news_sentiment'
        ],
        target=[target]
    )
    le = model.train()
    pred = [le[prediction] for prediction in (model.predict())]
    true = test.select(target)
    date = test.select('date')
    
    eval = pl.DataFrame(
        {'date': date,
         'true': true,
         'predicted': pred}
    )
    return eval, model.feature_importances()

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default='sp500')
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    currencies = 'United States(USD)'
    news = NewsSentiment(metric='median')
    sp500 = SP500Data()
    gold = GoldPriceData(currencies=[currencies])
    oil = OilPriceData()
    gap = 6

    args = parse_args()

    commodity = args.target
    target = '{}_price'.format(commodity)

    merged_dataset = MergedDataset(
        year=1960,
        gold=gold,
        oil=oil,
        sp500=sp500,
        news=news
    )

    merged_dataset.add_pct_changes(skip=commodity)
    merged_dataset.add_gains(skip=commodity)
    merged_dataset.add_moving_average(skip=commodity, window=5)
    #merged_dataset.add_future('sp500_price', gap=gap)
    merged_dataset.clean()

    print(merged_dataset.data)

    data = DataSet(
        data=merged_dataset.data,
        target_column=target,
        date_column='date',
        categorical=False,
        scale=True
        )
    
    X_train, X_test, y_train, y_test = data.split_data(test_size=0.25)

    dim = X_train.shape[1], X_train.shape[2]
    n_classes = merged_dataset.data.get_column(target).n_unique()
    lstm = RegressiveLSTM(dim)

    lstm.train(X_train, y_train, X_test, y_test)
    pred = lstm.predict(X_test)

    y_true = data.unscale(y_test)
    pred = data.unscale(pred)

    separator_index = len(merged_dataset.data)-len(pred)
    

    x1 = merged_dataset.data.get_column('date').to_list() # [separator_index:]
    y1 = data.unscale(y_train).tolist() + pred.tolist()
    y2 = data.unscale(y_train).tolist() + y_true.tolist()
    condition = np.array(y1) >= np.array(y2)

    plt.style.use('seaborn-pastel')
    plt.plot(x1, y1, label='prediction')
    plt.plot(x1, y2, label='true price')

    # Add a vertical line to separate train and test parts
    plt.axvline(x=x1[separator_index], color='gray', linestyle='--')
    
    # Fill the area between the two lines
    plt.fill_between(x1[:separator_index], y1[:separator_index], y2[:separator_index], color='skyblue', alpha=0.3)

    # Fill the area between the lines and the X-axis
    plt.fill_between(x1, y1, y2, where=condition, color='lightcoral', alpha=0.3, interpolate=True)
    plt.fill_between(x1, y1, y2, where=~condition, color='lightgreen', alpha=0.3, interpolate=True)

    plt.legend()
    plt.xlabel('year')
    plt.ylabel(f'{target} in {currencies}')
    plt.title('Prediction vs True Price')
    plt.show()
    exit()
    eval, importances = xgb(train, test)

    print(eval)

    sns.barplot(
        x=list(importances.keys()),
        y=list(importances.values()),
        palette='pastel')
    plt.show()

    dfm: pl.DataFrame = eval.melt('date', variable_name=target, value_name='signal')

    # sns.lineplot(data=dfm.to_pandas(), x='date', y='pct_change', hue=target, palette="pastel")
    sns.catplot(data=dfm.to_pandas(), x="date", y="signal", palette="pastel", hue=target, kind="swarm")
    plt.show()