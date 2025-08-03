from customerSatisfactionPrediction.configure import ConfigurationManager
from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.components import DataIngestion
from customerSatisfactionPrediction.components import DataValidation
from customerSatisfactionPrediction.components import DataTransformation
from customerSatisfactionPrediction.components import ModelTrainer
from customerSatisfactionPrediction.components import ModelEvaluation


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
    
    def model_trainer(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer_artifact = model_trainer.train()
    
    def model_evaluation(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
        