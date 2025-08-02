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

try:
    logger.info(f">>>>>>> Data Transformation Started <<<<<<<")
    data_transformation = TrainingPipeline()
    data_transformation.data_tranformation()
    logger.info(f">>>>>>> Data Transformation Completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>>>> Model Trainer Started <<<<<<<")
    model_trainer = TrainingPipeline()
    model_trainer.model_trainer()
    logger.info(f">>>>>>> Model Trainer Completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e