import datetime
import logging
import polars as pl
import seaborn as sns

from typing import Optional

from ..util.exceptions.exceptions import BadDateFormat

logger = logging.getLogger(__name__)

class OilPriceData:
    """
    A class that contains historic oil price information.
    """

    def __init__(self) -> None:
        
        # original data
        self.__raw_oil_price_data = pl.read_csv('data/oil/crude-oil-price.csv')

        # final data
        self.data: pl.DataFrame = self.__build_final_dataset()

    def __build_final_dataset(self):
        """
        Transforms the raw CSV data into a standardized format, which the other commodities also use.
        """
        dataset = self.__raw_oil_price_data.clone()
        dataset = dataset.with_columns([(pl.col("date").apply(lambda x: self.__normalize_dates(x)).alias('Normalized_Date'))])

        return dataset
    
    @staticmethod
    def __normalize_dates(datestring: str):
        """
        Turns the date time format used in the CSV into python datetime format.
        :paran str datestring: the string from the csv
        """
        year, month, day = tuple(datestring.partition('T')[0].split('-'))
        normalized = datetime.datetime(
            int(year),
            int(month),
            int(day))
        return normalized
    
    def plot(self):

        # make a lineplot
        plot = sns.lineplot(x="Normalized_Date", y="price", data=self.data.to_pandas())
        return plot
    
    def __find_closest_match(self, date: str):

        items = self.data['Normalized_Date'].to_list()
        closest_match = min(items, key=lambda x: abs(x - date))

        return closest_match

    def __getitem__(self, val) -> None:

        date = val

        # a string is passed
        if isinstance(date, str):
            try:
                dtm = [int(p) for p in date.split('-')]
                date = datetime.datetime(*dtm)

            except: # it is malformed
                raise BadDateFormat("It seems like you passed a string, but it isn't valid. It must follow YYYY-MM-DD format.")

        #elif isinstance(date, datetime._Date):
        #    pass
        #        
        #else:
        #    raise BadDateFormat('The date must be passed either as a string of the format YYYY-MM-DD or a datetime._Date object.')
        
        price = self.data.filter(pl.col('Normalized_Date') == date)['price']
        if len(price) > 0:  # there is an exact match
            price = price[0]
        
        else:  # otherwise find the nearest date
            date = self.__find_closest_match(date)
            price = self.data.filter(pl.col('Normalized_Date') == date)['price'][0]
        return price