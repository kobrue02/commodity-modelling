import datetime
import logging
import polars as pl
import torch

from datasets import Dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

from ..util.exceptions.exceptions import *
from ..price_data import HistoricPriceData

logger = logging.getLogger(__name__)

class NewsSentiment(HistoricPriceData):
    """
    A class that contains historic news sentiment.
    """

    def __init__(self,
                 from_scratch: bool = False,
                 filename: str = None,
                 device: str = 'cuda:0',
                 batch_size: int = 8,
                 trunctuation: str = 'only_first',
                 metric: str = 'median'
                 ) -> None:
        
        super().__init__()

        self.metric = metric
        if from_scratch:  # calculate sentiment from scratch

            # original data
            self.__abc_news_articles = pl.read_csv(filename)

            # nlp pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                device=device,
                batch_size=batch_size,
                truncation=trunctuation
                )
            # final data
            self.raw_data: pl.DataFrame = self.__annotate_dataset()
        else:
            self.raw_data: pl.DataFrame = pl.read_csv('data/news/news_with_sentiment.csv')
        
        self.data = self.__build_final_dataset()

    def save(self):
        self.raw_data.write_csv('data/news/news_with_sentiment.csv')

    def __annotate_dataset(self) -> pl.DataFrame:
        dataset = self.__abc_news_articles.clone()
        df = Dataset.from_pandas(dataset.to_pandas())

        sentiment_lst = []
        for out in tqdm(self.sentiment_pipeline(KeyDataset(df, "headline_text")), total=len(df)):
            sentiment = self.__get_sentiment(out)
            sentiment_lst.append(sentiment)

        return dataset.with_columns(pl.Series(name="sentiment", values=sentiment_lst)) 

    def __get_sentiment(self, sentiment: dict) -> float:

        if sentiment['label'] == 'POSITIVE':
            return sentiment['score']
        else:
            return 0 - sentiment['score']

    def __getitem__(self, val):
        date = val
        return self.getval(date, 'sentiment')
    
    def __build_final_dataset(self) -> pl.DataFrame:

        dataset = self.raw_data.with_columns([(pl.col("publish_date").apply(lambda x: self.normalize_dates(str(x))).alias('Normalized_Date'))])

        output_data_dict = {
            'Normalized_Date': [],
            'sentiment': []
        }

        dataset = dataset.with_columns(
            pl.col("Normalized_Date").dt.year().alias('year'),
            pl.col("Normalized_Date").dt.month().alias('month')
            )
        
        for year in range(min(dataset['year'].to_list()), max(dataset['year'].to_list())):
            year_df = dataset.filter(pl.col("year") == year)
            for month in range(min(year_df['month'].to_list()), max(year_df['month'].to_list())):

                # get the metric (e.g. median) of the month
                if self.metric == 'median':
                    metric = year_df.filter(pl.col('month') == month)['sentiment'].median()
                elif self.metric == 'mean':
                    metric = year_df.filter(pl.col('month') == month)['sentiment'].mean()
                else:
                    raise UnsupportedMetric('Only median and mean are supported.')

                output_data_dict['Normalized_Date'].append(
                    datetime.datetime(
                        year,
                        month,
                        15
                    )
                )
                output_data_dict['sentiment'].append(metric)

        output_data_df = pl.DataFrame(output_data_dict)
        return output_data_df