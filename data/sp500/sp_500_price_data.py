import logging
import polars as pl

from ..price_data import HistoricPriceData

logger = logging.getLogger(__name__)

class SP500Data(HistoricPriceData):
    """
    A class that contains historic oil price information.
    """

    def __init__(self) -> None:
        
        super().__init__()

        # original data
        self.__raw_oil_price_data = pl.read_csv('data/sp500/SPX.csv')

        # final data
        self.data: pl.DataFrame = self.__build_final_dataset()

    def __build_final_dataset(self):
        """
        Transforms the raw CSV data into a standardized format, which the other commodities also use.
        """
        dataset = self.__raw_oil_price_data.clone()
        dataset = dataset.with_columns([(pl.col("Date").apply(lambda x: self.normalize_dates(x)).alias('Normalized_Date'))])

        return dataset

    def __getitem__(self, val) -> None:
        
        col = 'Close'
        date = val

        return self.getval(date, col)