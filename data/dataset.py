import datetime
import logging
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from dataclasses import dataclass

from dateutil.relativedelta import relativedelta
from typing import Optional

from data.util import price_data_item
from data.util.exceptions.exceptions import *
from data.gold.gold_price_data import GoldPriceData
from data.oil.oil_price_data import OilPriceData
from data.sp500.sp_500_price_data import SP500Data
from data.news.news_sentiment import NewsSentiment

logger = logging.getLogger(__name__)

class MergedDataset:
    """
    Merge datasets of different commodities and economic factors into one large polars DataFrame.
    """
    def __init__(self, year: Optional[int] = None, **kwargs) -> None:
        
        ALLOWED_KEYS = {'gold', 'oil', 'sp500', 'news'}
        supported_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED_KEYS}

        # user args
        self.__year = year

        # unpacking keyword arguments
        self.__dict__.update(supported_kwargs)
        self.__factors = list(supported_kwargs.keys())

        # class attrs
        self.commodities = ['gold', 'oil', 'sp500']
        self.econ_factors = ['news']

        # output
        self.data = self.__build_merged_dataset()

    def __find_initial_date(self):
        dates = [
            getattr(self, factor).data['Normalized_Date'].to_list()[0] for factor in self.__factors
        ]

        if not self.__year:
            return max(dates)
        else:
            user_date = datetime.datetime(
                self.__year,
                1,
                1)
            if user_date > max(dates):
                return user_date
            else:
                logger.warning('{} is out of range, starting at {} instead'.format(self.__year, str(max(dates))))
                return max(dates)

    def __build_merged_dataset(self) -> pl.DataFrame:
        """
        Merge all given datasets into one by using the date column as a common index.
        """
        prices = {}
        initial_date = self.__find_initial_date()

        # the values for each column will be collected in lists
        # get the value of each dataset at the initial date
        for factor in self.__factors:

            if factor in self.commodities:
                prices[f'{factor}_price'] = []
                prices[f'{factor}_price'].append(getattr(self, factor)[initial_date])
            elif factor in self.econ_factors:
                prices[f'{factor}_sentiment'] = []
                prices[f'{factor}_sentiment'].append(getattr(self, factor)[initial_date])

        date_lst = []
        date_lst.append(initial_date)
        
        current_date = initial_date

        # iterate over each month
        for i in range(len(self.oil.data['Normalized_Date'].to_list())):
            
            current_prices = {}
            current_date = current_date + relativedelta(months=1)  # add 1 month

            try:
                for factor in self.__factors:
                    current_prices[factor] = getattr(self, factor)[current_date]

            # once the end of one dataset is reached, break loop
            except DateNotInDataError:
                break

            for factor in self.__factors:
                if factor in self.commodities:
                    prices[f'{factor}_price'].append(current_prices[factor])
                elif factor in self.econ_factors:
                    prices[f'{factor}_sentiment'].append(current_prices[factor])
            
            date_lst.append(current_date)

        # add all info in a dict which polars can interpret
        merged_dataset_dict = {
            'date': date_lst
        }
        merged_dataset_dict.update(prices)

        return pl.from_dict(merged_dataset_dict)
    
    def add_pct_changes(self, inplace: bool = True, skip: str = None):
        for factor in self.__factors:
            if 'news' in factor or factor == skip:
                continue
            col_name = f'{factor}_price'

            if inplace:
                self.data = self.data.with_columns(pl.col(col_name).pct_change().alias("{}_pct_change".format(factor))).fill_null(strategy='zero')
            else:
                df = self.data.clone()
                return df.with_columns(pl.col(col_name).pct_change().alias("{}_pct_change".format(factor))).fill_null(strategy='zero')
    
    def add_gains(self, inplace: bool = True, skip: str = None):

        for factor in self.__factors:
            if factor == skip or factor == 'news':
                continue
            if factor in self.commodities:
                col_name = f'{factor}_price'

            initial_value: float = self.data[col_name].to_list()[0]

            if inplace:
                self.data = self.data.with_columns([(pl.col(col_name).apply(lambda x: self.__pct_gains(initial_value, x)).alias(f'{factor}_gains'))])
            else:
                df = self.data.clone()
                return df.with_columns([(pl.col(col_name).apply(lambda x: self.__pct_gains(initial_value, x)).alias(f'{factor}_gains'))])

    def __add_future(self, date, future_gap: int, col: str):
        old_date = date
        future_date = date + relativedelta(months=future_gap)

        old_price = self.data.filter(pl.col('date') == old_date).select(col).item()

        try:
            future_price = self.data.filter(pl.col('date') == future_date).select(col).item()
        except ValueError:
            return None
        
        diff = future_price - old_price

        if diff > 50:
            signal = 'buy'
        elif -50 < diff <= 50:
            signal = 'hold'
        else:
            signal = 'sell'
        return signal
    
    def add_future(self, column: str = 'sp500_price', gap: int = 6, inplace: bool = True):
        if inplace:
            self.data = self.data.with_columns(
                [(pl.col('date').apply(lambda x: self.__add_future(
                    x,
                    future_gap=gap,
                    col=column
                    )).alias(f'{column}_signal'))])
        else:
            return self.data.with_columns(
                    [(pl.col('date').apply(lambda x: self.__add_future(
                        x,
                        future_gap=gap,
                        col=column
                        )).alias(f'{column}_signal'))])
    
    def add_moving_average(self, skip: str, window: int = 50):
        for factor in self.__factors:
            if factor == skip or factor == 'news':
                continue
            # N-day moving average
            self.data = self.data.with_columns([(pl.col(f'{factor}_price').rolling_mean(window_size=window).alias(f'{factor}_MA_{window}'))])
            self.data = self.data.with_columns([(pl.col(f'{factor}_price').rolling_mean(window_size=window).alias(f'{factor}_MA_{window}'))])

    def __pct_gains(self, initial: float, current: float) -> float:
        return (current - initial) / initial * 100

    def plot(self, kind: str = 'price'):
        # prepare the data for plotting
        dfm_commodities = self.__transform_data_for_plot(self.data, kind)
        dfm_econ = self.__transform_data_for_plot(self.data, kind='sentiment')

        # lets make a lineplot

        # left y-axis (commodity prices)
        plot = sns.lineplot(
            x="date",
            y=kind,
            hue='Commodity',
            data=dfm_commodities.to_pandas(),
            palette="flare"
            )
        plot.set(xlabel="Date", ylabel="Gains in %", title='Performance of different investments over time')

        # right y-axis (news)
        ax2 = plt.twinx()
        sns.lineplot(
            data=dfm_econ.to_pandas(),
            x="date",
            y="sentiment",
            hue="Sentiment",
            ax=ax2,
            palette="pastel"
            )
        return plot
    
    def __transform_data_for_plot(self, data: pl.DataFrame, kind: str) -> pl.DataFrame:

        if kind != 'sentiment':
            cols = [col for col in data.columns if kind in col and 'news' not in col]
        else:
            cols = ['news_sentiment']
        cols = ["date"] + cols

        df: pl.DataFrame = data[cols].clone()
        df = df.rename(lambda column_name: column_name.split('_')[0])

        if kind != 'sentiment':
            dfm: pl.DataFrame = df.melt('date', variable_name='Commodity', value_name=kind)
        else:
            dfm: pl.DataFrame = df.melt('date', variable_name='Sentiment', value_name=kind)
        
        return dfm

    def clean(self):
        self.data = self.data.select(pl.all().interpolate())
        self.data = self.data.drop_nulls()

if __name__ == "__main__":

    news = NewsSentiment(metric='median')
    sp500 = SP500Data()
    gold = GoldPriceData(currencies=['United States(USD)'])
    oil = OilPriceData()

    merged_dataset = MergedDataset(
        year=1990,
        gold=gold,
        oil=oil,
        sp500=sp500,
        news=news
    )

    # merged_dataset.add_pct_changes()
    merged_dataset.add_gains()

    print(merged_dataset.data.tail())

    plot = merged_dataset.plot(kind='gains')
    plt.show()
