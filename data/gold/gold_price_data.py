import datetime
import logging
import polars as pl
import seaborn as sns

from typing import Optional

from ..util.exceptions.exceptions import BadDateFormat, DateNotInDataError

logger = logging.getLogger(__name__)

class GoldPriceData:
    """
    A class which loads the historic price data of Gold from 1979 to 2021 in various currencies.
    """

    def __init__(self,
                 currencies: list[str] = [
                     'United States(USD)',
                     'Europe(EUR)',
                     'Japan(JPY)'
                 ],
                 date_range: Optional[tuple[str]] = None,
                 **kwargs,
                 ) -> None:
        """
        Initializes a GoldPriceData object, which contains all the data in standardized format, prepares plots, and more.

        :param list currencies: a list of currencies the user wants to draw
        :param tuple date_range: a tuple with start and end date specified like this: year_month
        """

        # user arguments
        self._currencies = currencies
        self._date_range = date_range
        
        # unpacking keyword arguments
        self._kind = kwargs.get('kind', 'monthly')
        self._tightness = kwargs.get('tightness', 'loose')

        # original data
        self.__raw_gold_price_data = pl.read_csv('data/gold/1979-2021.csv')

        # output
        self.data: pl.DataFrame = self.__build_final_dataset()
    
    @staticmethod
    def __normalize_dates(datestring: str):
        """
        Turns the date time format used in the CSV into python datetime format.
        :paran str datestring: the string from the csv
        """
        day, month, year = tuple(datestring.split('-'))
        normalized = datetime.datetime(
            int(year),
            int(month),
            int(day))
        return normalized
    
    def __build_final_dataset(self):
        """
        Transforms the raw CSV data into a standardized format, which the other commodities also use.
        """
        dataset = self.__raw_gold_price_data.clone()
        dataset = dataset.with_columns([(pl.col("Date").apply(lambda x: self.__normalize_dates(x)).alias('Normalized_Date'))])

        if self._kind == 'annual':
            ## TODO
            pass

        return dataset[['Normalized_Date'] + self._currencies]

    def __transform_data_for_plot(self):

        dfm: pl.DataFrame = self.data.melt('Normalized_Date', variable_name='Currency', value_name='vals')
        return dfm
    
    def plot(self):
        
        # prepare the data for plotting
        dfm = self.__transform_data_for_plot()

        # make a lineplot
        plot = sns.lineplot(x="Normalized_Date", y="vals", hue='Currency', data=dfm.to_pandas())
        return plot

    def __find_closest_match(self, date):

        items = self.data['Normalized_Date'].to_list()
        closest_match = min(items, key=lambda x: abs(x - date))
        
        diff = date - closest_match
        if diff.days < 50:
            return closest_match
        else:
            raise DateNotInDataError('No datapoint near {} was found.'.format(str(date)))

    def set_tightness(self, tightness: str = 'tight'):
        self._tightness = tightness

    def __getitem__(self, val) -> None:

        if isinstance(val, tuple):
            date, currency = val
        else:
            date = val
            currency = self._currencies[0]

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
        
        price = self.data.filter(pl.col('Normalized_Date') == date)[currency]
        if len(price) > 0:  # there is an exact match
            price = price[0]
        
        else:  # otherwise find the nearest date
            if self._tightness == 'loose':
                date = self.__find_closest_match(date)
                price = self.data.filter(pl.col('Normalized_Date') == date)[currency][0]

            if self._tightness == 'tight':
                return None
        return price
        