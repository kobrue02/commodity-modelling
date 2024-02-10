import datetime
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from dateutil.relativedelta import relativedelta

from models.models import XGBRegressor
from data.dataset import MergedDataset
from data.util.exceptions.exceptions import *
from data.gold.gold_price_data import GoldPriceData
from data.oil.oil_price_data import OilPriceData
from data.sp500.sp_500_price_data import SP500Data
from data.news.news_sentiment import NewsSentiment


if __name__ == "__main__":

    news = NewsSentiment(metric='median')
    sp500 = SP500Data()
    gold = GoldPriceData(currencies=['United States(USD)'])
    oil = OilPriceData()
    gap = 6

    merged_dataset = MergedDataset(
        year=1960,
        gold=gold,
        oil=oil,
        sp500=sp500
    )

    trg = 'gold_price'

    merged_dataset.add_pct_changes()
    merged_dataset.add_gains()
    merged_dataset.add_future(trg, gap=gap)
    merged_dataset.clean()

    print(merged_dataset.data)

    cutoff = datetime.datetime(2018, 8, 15)
    train = merged_dataset.data.filter(pl.col('date') < cutoff)
    test = merged_dataset.data.filter(pl.col('date') >= cutoff)

    target = f'{trg}_after_{gap}_months'

    model = XGBRegressor(
        train_set=train,
        test_set=test,
        features=[
            'sp500_price',
            'gold_price',
            'oil_price'
        ],
        target=[target]
    )
    model.train()
    pred = model.predict()
    true = test.select(target)
    date = test.select('date')
    
    eval = pl.DataFrame(
        {'date': date,
         'true': true,
         'predicted': pred}
    )

    print(eval)
    dfm: pl.DataFrame = eval.melt('date', variable_name=target, value_name='pct_change')
    sns.lineplot(data=dfm.to_pandas(), x='date', y='pct_change', hue=target, palette="pastel")
    plt.show()