import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from dataclasses import dataclass

from dateutil.relativedelta import relativedelta
from typing import Optional

from data.util import price_data_item
from data.util.exceptions.exceptions import DateNotInDataError
from data.gold.gold_price_data import GoldPriceData
from data.oil.oil_price_data import OilPriceData
from data.sp500.sp_500_price_data import SP500Data

class MergedDataset:
    """
    Merge datasets of different commodities and economic factors into one large polars DataFrame.
    """
    def __init__(self, **kwargs) -> None:
        
        ALLOWED_KEYS = {'gold', 'oil', 'sp500'}
        supported_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED_KEYS}

        # unpacking keyword arguments
        self.__dict__.update(supported_kwargs)
        self.__factors = list(supported_kwargs.keys())

        # output
        self.data = self.__build_merged_dataset()

    def __build_merged_dataset(self) -> pl.DataFrame:
        """
        Merge all given datasets into one by using the date column as a common index.
        """
        prices = {}
        initial_date = self.oil.data['Normalized_Date'].to_list()[0]  # fix later

        # the values for each column will be collected in lists
        # get the value of each dataset at the initial date
        for factor in self.__factors:
            prices[f'{factor}_price'] = []
            prices[f'{factor}_price'].append(getattr(self, factor)[initial_date])

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
                prices[f'{factor}_price'].append(current_prices[factor])
            
            date_lst.append(current_date)

        # add all info in a dict which polars can interpret
        merged_dataset_dict = {
            'date': date_lst
        }
        merged_dataset_dict.update(prices)

        return pl.from_dict(merged_dataset_dict)
    
    def add_pct_changes(self, inplace: bool = True):
        for factor in self.__factors:
            col_name = f'{factor}_price'

            if inplace:
                self.data = self.data.with_columns(pl.col(col_name).pct_change().alias("{}_pct_change".format(factor)))
            else:
                df = self.data.clone()
                return df.with_columns(pl.col(col_name).pct_change().alias("{}_pct_change".format(factor)))
    
    def add_gains(self, inplace: bool = True):

        for factor in self.__factors:
            col_name = f'{factor}_price'
            initial_value: float = self.data[col_name].to_list()[0]

            if inplace:
                self.data = self.data.with_columns([(pl.col(col_name).apply(lambda x: self.__pct_gains(initial_value, x)).alias(f'{factor}_gains'))])
            else:
                df = self.data.clone()
                return df.with_columns([(pl.col(col_name).apply(lambda x: self.__pct_gains(initial_value, x)).alias(f'{factor}_gains'))])

    def __pct_gains(self, initial: float, current: float) -> float:
        return (current - initial) / initial * 100

    def plot(self, kind: str = 'price'):
        # prepare the data for plotting
        dfm = self.__transform_data_for_plot(kind)

        # make a lineplot
        plot = sns.lineplot(x="date", y=kind, hue='Commodity', data=dfm.to_pandas())
        plot.set(xlabel="Date", ylabel="Gains in %", title='Performance of different investments over time')
        return plot
    
    def __transform_data_for_plot(self, kind: str) -> pl.DataFrame:

        cols = [col for col in self.data.columns if kind in col]
        cols = ["date"] + cols

        df: pl.DataFrame = self.data[cols].clone()
        df = df.rename(lambda column_name: column_name.split('_')[0])

        dfm: pl.DataFrame = df.melt('date', variable_name='Commodity', value_name=kind)
        
        return dfm




if __name__ == "__main__":

    sp500 = SP500Data()
    gold = GoldPriceData(currencies=['United States(USD)'])
    oil = OilPriceData()

    merged_dataset = MergedDataset(
        gold = gold,
        oil = oil,
        sp500 = sp500
    )

    # merged_dataset.add_pct_changes()
    merged_dataset.add_gains()

    print(merged_dataset.data.head())

    plot = merged_dataset.plot(kind='gains')
    plt.show()
