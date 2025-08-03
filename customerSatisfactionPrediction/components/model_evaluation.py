import sys
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.exceptions import CSPException
from customerSatisfactionPrediction.entity import ModelEvaluationConfig
from customerSatisfactionPrediction.utils import load_bin, save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average="weighted")
        recall = recall_score(actual, pred, average="weighted")
        f1 = f1_score(actual, pred, average="weighted")
        confusion = confusion_matrix(actual, pred)
        return accuracy, precision, recall, f1, confusion
    
    def initiate_model_evaluation(self):
        try:
            test_data = pd.read_csv(self.config.test_data_path, header= None)
            model = load_bin(Path(self.config.model_path))

            test_x = test_data.iloc[:, :-1]
            test_x = test_x.iloc[:, :5000]
            test_y = test_data.iloc[:, -1]

            predicted_qualities = model.predict(test_x)
            (accuracy, precision, recall, f1, confusion) = self.eval_metrics(test_y, predicted_qualities)

            # Saving Metrics as Local
            scores = {
                "Accuracy Score": accuracy,
                "Precision Score": precision,
                "Recall Score": recall,
                "F1 Score": f1,
                "Confusion Metrix": confusion.tolist()
            }
            save_json(path = Path(self.config.metric_file_name), data=scores)

            logger.info(f" Model Evaluation Complated. Metrics Saved to {self.config.metric_file_name}")
        except Exception as e:
            logger.error(f"Error in Model Evaluation: {str(e)}")
            raise CSPException(e, sys)