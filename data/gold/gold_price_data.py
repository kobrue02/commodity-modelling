import datetime
import logging
import polars as pl
import seaborn as sns

from typing import Optional

from ..util.exceptions.exceptions import BadDateFormat, DateNotInDataError
from ..price_data import HistoricPriceData

logger = logging.getLogger(__name__)

class GoldPriceData(HistoricPriceData):
    """
    A class which loads the historic price data of Gold from 1979 to 2021 in various currencies.
    """

    def __init__(self,
                 currencies: list[str] = [
                     'United States(USD)',
                     'Europe(EUR)',
                     'Japan(JPY)'
                 ],
                 **kwargs,
                 ) -> None:
        """
        Initializes a GoldPriceData object, which contains all the data in standardized format, prepares plots, and more.

        :param list currencies: a list of currencies the user wants to draw
        :param tuple date_range: can be a string "all" for the whole possible time frame or a tuple with start and end date specified like this: year_month
        """

        # initialize super class
        super().__init__(**kwargs)

        # user arguments
        self._currencies = currencies

        # original data
        self.__raw_gold_price_data = pl.read_csv('data/gold/1979-2021.csv')

        # output
        self.data: pl.DataFrame = self.__build_final_dataset()
    
    def __build_final_dataset(self):
        """
        Transforms the raw CSV data into a standardized format, which the other commodities also use.
        """
        dataset = self.__raw_gold_price_data.clone()
        dataset = dataset.with_columns([(pl.col("Date").apply(lambda x: self.normalize_dates(x, 'dmy')).alias('Normalized_Date'))])

        if self._kind == 'annual':
            ## TODO
            pass

        return dataset[['Normalized_Date'] + self._currencies]

    def set_tightness(self, tightness: str = 'tight'):
        self._tightness = tightness

    def __getitem__(self, val) -> None:

        if isinstance(val, tuple):
            date, currency = val
        else:
            date = val
            currency = self._currencies[0]

        return self.getval(date, currency)
        