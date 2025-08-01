from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.pipeline import TrainingPipeline

try:
    logger.info(f">>>>>>> Data Ingestion Started <<<<<<<")
    data_ingestion = TrainingPipeline()
    data_ingestion.data_ingestion()
    logger.info(f">>>>>>> Data Ingestion Completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>>> Data Validation Started <<<<<<<")
    data_validation = TrainingPipeline()
    data_validation.data_validation()
    logger.info(f">>>>>>> Data Validation Completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e