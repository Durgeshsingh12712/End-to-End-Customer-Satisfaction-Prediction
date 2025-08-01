import pandas as pd

from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = [
                "Ticket ID", "Customer Name", "Customer Email", "Customer Age", 
                "Customer Gender", "Product Purchased", "Date of Purchase", 
                "Ticket Type", "Ticket Subject", "Ticket Description", 
                "Ticket Status", "Resolution", "Ticket Priority", 
                "Ticket Channel", "First Response Time", "Time to Resolution", 
                "Customer Satisfaction Rating"
            ]

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation Status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation Status: {validation_status}")
            
            return validation_status
        except Exception as e:
            raise e
    
    def validate_data_types(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir)

            critical_columns = ["Ticket Type", "Ticket Subject", "Ticket Description", "Ticket Priority"]
            missing_critical = data[critical_columns].isnull().sum().sum()

            if missing_critical > 0:
                logger.warning(f"Missing values found in critical columns: {missing_critical}")
                return False
            
            expected_dtypes = {
                "Ticket ID": "int64",
                "Customer Age": "int64"
            }

            for col, dtype in expected_dtypes.items():
                if str(data[col].dtype) != dtype:
                    logger.error(f"Column {col} has encorrectly dtype: {data[col].dtype}, expected: {dtype}")
                    return False
                
            logger.info("Data Validation passed successfully ")
            return True
        except Exception as e:
            logger.error(f"Error in Data Validation: {str(e)}")
            raise e
    
    def initiate_data_validation(self):
        self.validate_all_columns()
        self.validate_data_types()