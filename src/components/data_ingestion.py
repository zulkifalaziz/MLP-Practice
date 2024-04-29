import os
import sys
import pandas as pd
from src.exception import CustomError
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data = os.path.join('..\\artifacts', 'train.csv')
    test_data = os.path.join('..\\artifacts', 'test.csv')
    raw_data = os.path.join('..\\artifacts','data.csv')

class DataInjestion:
    def __init__(self):
        self.injestion_config = DataIngestionConfig()
        pass

    def injestion(self):
        try:
            df = pd.read_csv("C:\\Users\\zulki\\Desktop\\archive\\study.csv")
            logging.info('Successfully read the dataset as a dataframe!')

            os.makedirs(os.path.dirname(self.injestion_config.train_data), exist_ok=True)
            df.to_csv(self.injestion_config.raw_data, index=False, header=True)
            logging.info('Train test split initiated')
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            test_data.to_csv(self.injestion_config.test_data, index=False, header=True)
            train_data.to_csv(self.injestion_config.train_data, index=False, header=True)
            logging.info('Successfully created train test data')
            return (
                self.injestion_config.train_data,
                self.injestion_config.test_data
            )
        except Exception as e:
            logging.info('Error Raised!')
            raise CustomError(e, sys)
        pass


obj = DataInjestion()
x = obj.injestion()
print(x)