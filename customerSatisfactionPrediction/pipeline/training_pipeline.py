from customerSatisfactionPrediction.configure import ConfigurationManager
from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.components import DataIngestion
from customerSatisfactionPrediction.components import DataValidation
from customerSatisfactionPrediction.components import DataTransformation


class TrainingPipeline:
    def __init__(self):
        pass

    def data_ingestion(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    def data_validation(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
    
    def data_tranformation(self):
        config = ConfigurationManager()
        data_tranformation_config = config.get_data_transformation_config()
        data_tranformation = DataTransformation(config=data_tranformation_config)
        data_tranformation_artifact = data_tranformation.initiate_data_transformation()