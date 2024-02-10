import datetime
import logging
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from dataclasses import dataclass

from dateutil.relativedelta import relativedelta
from typing import Optional

from .util.exceptions.exceptions import DateNotInDataError, BadDateFormat

logger = logging.getLogger(__name__)

class HistoricPriceData:


    def __init__(self, **kwargs) -> None:

        # unpacking kwargs
        default_args = {
            '_kind': 'monthly',
            '_tightness': 'loose',
            '_date_range': 'all'
            }
    
        supported_kwargs = {f"_{k}": v for k, v in kwargs.items() if f"_{k}" in default_args}
        supported_kwargs.update({k: v for k, v in default_args.items() if k not in supported_kwargs})

        # unpacking keyword arguments
        self.__dict__.update(supported_kwargs)

        # abstract variables
        self.data: pl.DataFrame


    def find_closest_match(self, date):

        items = self.data['Normalized_Date'].to_list()
        closest_match = min(items, key=lambda x: abs(x - date))
        
        diff = date - closest_match
        if diff.days < 50:
            return closest_match
        else:
            raise DateNotInDataError('No datapoint near {} was found.'.format(str(date)))
        
    @staticmethod
    def normalize_dates(datestring: str, src: str = 'ymd'):
        """
        Turns the date time format used in the CSV into python datetime format.
        :paran str datestring: the string from the csv
        """
        if src == 'dmy':
            day, month, year = tuple(datestring.split('-'))
        elif src == 'ymd':
            year, month, day = tuple(datestring.partition('T')[0].split('-'))
        else:
            raise BadDateFormat('The date must be either in DD-MM-YYYY or YYYY-MM-DD format.')
        
        normalized = datetime.datetime(
            int(year),
            int(month),
            int(day))
        return normalized
    
    def transform_data_for_plot(self):

        dfm: pl.DataFrame = self.data.melt('Normalized_Date', variable_name='Currency', value_name='price')
        return dfm
    
    def plot(self, y_column: str, normalize_data: bool = False):

        if normalize_data:
            # prepare the data for plotting
            data = self.transform_data_for_plot()
        else:
            data = self.data.clone()

        # make a lineplot
        plot = sns.lineplot(x="Normalized_Date", y=y_column, data=data.to_pandas())
        return plot
    
    def getval(self,
               val,
               column: str = 'price',
               tightness: Optional[str] = 'loose'
               ) -> None:

        """
        Super method to help sub classes implement the __getitem__() method.

        :param str val: the datetime in either string format or datetime._Date
        :param str column: the name of the column which we ar interested in, e.g. 'price'
        :param str tightness: whether to only return exact date matches or closest match
        """
        date = val

        # a string is passed
        if isinstance(date, str):
            try:
                dtm = [int(p) for p in date.split('-')]
                date = datetime.datetime(*dtm)

            except: # it is malformed
                raise BadDateFormat("It seems like you passed a string, but it isn't valid. It must follow YYYY-MM-DD format.")
        
        price = self.data.filter(pl.col('Normalized_Date') == date)[column]

        if len(price) > 0:  # there is an exact match
            price = price[0]
        
        else:  # otherwise find the nearest date
            if tightness == 'loose':
                date = self.find_closest_match(date)
                price = self.data.filter(pl.col('Normalized_Date') == date)[column][0]

            if tightness == 'tight':
                return None
        return price