import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from dataclasses import dataclass

from dateutil.relativedelta import relativedelta
from typing import Optional

from data.util import price_data_item
from data.gold.gold_price_data import GoldPriceData
from data.oil.oil_price_data import OilPriceData

class MergedDataset:
    def __init__(self, **kwargs) -> None:

        # unpacking keyword arguments
        self.gold: GoldPriceData = kwargs.get('gold', None)
        self.oil: OilPriceData = kwargs.get('oil', None)

        # output
        self.data = self.__build_merged_dataset()

    def __build_merged_dataset(self) -> pl.DataFrame:
        date_lst = []
        oil_lst = []
        gold_lst = []

        initial_date = self.oil.data['Normalized_Date'].to_list()[0]  # fix later
        date_lst.append(initial_date)

        initial_oil_price = self.oil[initial_date]
        oil_lst.append(initial_oil_price)

        initial_gold_price = self.gold[initial_date]
        gold_lst.append(initial_gold_price)

        current_date = initial_date
        for i in range(len(self.oil.data['Normalized_Date'].to_list())):

            current_date = current_date + relativedelta(months=1)
            current_oil_price = self.oil[current_date]
            current_gold_price = self.gold[current_date]

            date_lst.append(current_date)
            oil_lst.append(current_oil_price)
            gold_lst.append(current_gold_price)

        merged_dataset_dict = {
            'date': date_lst,
            'gold': gold_lst,
            'oil': oil_lst
        }

        return pl.from_dict(merged_dataset_dict)
    
    def plot(self):
        # prepare the data for plotting
        dfm = self.__transform_data_for_plot()

        # make a lineplot
        plot = sns.lineplot(x="date", y="Price", hue='Commodity', data=dfm.to_pandas())
        return plot
    
    def __transform_data_for_plot(self):

        dfm: pl.DataFrame = self.data.melt('date', variable_name='Commodity', value_name='Price')
        return dfm




if __name__ == "__main__":
    
    gold = GoldPriceData(currencies=['United States(USD)'])
    oil = OilPriceData()

    merged_dataset = MergedDataset(
        gold = gold,
        oil = oil
    )

    print(merged_dataset.data.head())
    plot = merged_dataset.plot()
    plt.show()
