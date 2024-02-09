import matplotlib.pyplot as plt
import polars as pl

from dataclasses import dataclass

from data.util import price_data_item
from data.gold.gold_price_data import GoldPriceData


#oil_price_data = pl.read_csv('data/oil/crude-oil-price.csv')
#inflation_data = pl.read_csv('data/inflation/inflation.csv')

class OilPriceData:
    def __init__(self) -> None:
        pass

class InflationData:
    def __init__(self) -> None:
        pass

if __name__ == "__main__":
    gold = GoldPriceData(kind='annual')


    print(gold.data.head())
    plot = gold.plot()
    plt.show()

    
    print(gold['1980-01-31', 'United States(USD)'])