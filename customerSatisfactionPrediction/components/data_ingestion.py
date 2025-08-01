import os, zipfile
import pandas as pd
import urllib.request as request
from pathlib import Path

from sklearn.model_selection import train_test_split

from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.entity import DataIngestionConfig
from customerSatisfactionPrediction.utils import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, header = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{header}")
        else:
            logger.info(f"File Already exists of size: {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    
    def initiate_data_ingestion(self):
        try:
            self.download_file()

            df = pd.read_csv(self.config.local_data_file)
            logger.info(f"Dataset Shape: {df.shape}")

            train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Ticket Priority'])

            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")

            train.to_csv(train_path, index = False)
            test.to_csv(test_path, index = False)

            logger.info(f"Train Data Saved to: {train_path}")
            logger.info(f"Test Data Saved to: {test_path}")

            return train_path, test_path
        
        except Exception as e:
            raise e
