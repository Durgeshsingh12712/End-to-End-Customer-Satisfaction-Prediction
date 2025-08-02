import os, sys, warnings
import pandas as pd
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.exceptions import CSPException
from customerSatisfactionPrediction.entity import ModelTrainerConfig
from customerSatisfactionPrediction.utils import save_bin, save_json

warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def get_models(self):  
        """Return dict with all classifiers and tuned hyper-parameters"""
        try:
            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    max_depth=20,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    bootstrap=True,
                ),

                "LogisticRegression": Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                penalty="l2",
                                C=1.0,
                                solver="lbfgs",
                                max_iter=1000,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ]
                ),

                "KNearestNeighbors": Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            KNeighborsClassifier(
                                n_neighbors=5,
                                weights="distance",
                                algorithm="auto",
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            }
            return models
        except Exception as e:
            raise CSPException(e, sys)
    
    def preprocess_features(self, train_x, test_x, train_y):
        """Feature Selection + SMOTE balancing (Only Train)"""
        try:
            logger.info("Applying Feature Preprocessing")

            # 2-A Feature Selection (chi^2)
            k_features = min(5000, train_x.shape[1])
            if train_x.shape[1] > k_features:
                logger.info(f"Selection Top {k_features} of {train_x.shape[1]} Features (chi^2)")
                
                selector = SelectKBest(chi2, k=k_features)
                train_x_sel = selector.fit_transform(train_x, train_y)
                test_x_sel = selector.transform(test_x)

                # Persist Selector for instance stage
                selector_path = os.path.join(self.config.root_dir, "feature_selector.pkl")
                save_bin(selector, Path(selector_path))

                train_x = pd.DataFrame(train_x_sel)
                test_x = pd.DataFrame(test_x_sel)
                logger.info(f"Feature Selection Done. Shape : {train_x.shape}")
            
            # 2-B Class re-sampling with SMOTE
            logger.info("Applying SMOTE on Training Data")
            smote = SMOTE(random_state=42, k_neighbors=3)
            train_x_bal, train_y_bal = smote.fit_resample(train_x, train_y)
            logger.info(f"SMOTE complete - before {train_x.shape}, after {train_x_bal.shape}")

            return train_x_bal, test_x, train_y_bal
        
        except Exception as e:
            logger.warning(f"Feature Preprocessing Failed: {str(e)} - using raw data.")
            return train_x, test_x, train_y
    
    def train(self):
        try:
            logger.info("Model Training Started")

            train_df = pd.read_csv(self.config.train_data_path, header=None)
            test_df = pd.read_csv(self.config.test_data_path, header=None)

            train_x, train_y = train_df.iloc[:, :-1], train_df.iloc[:, -1]
            test_x, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1]

            logger.info(f"Loaded Data -> Train {train_x.shape}, Test {test_x.shape}")
            
            train_x, test_x, train_y = self.preprocess_features(train_x, test_x, train_y)

            models = self.get_models()

            best_model, best_score, best_name = None, 0.0, ""
            model_scores = {}

            for name, model in models.items():
                logger.info(f"Training {name} ...")

                try:
                    # REtrain on full Training Data
                    model.fit(train_x, train_y)

                    # Evaluate on Train & Test
                    train_pred = model.predict(train_x)
                    test_pred = model.predict(test_x)

                    train_acc = accuracy_score(train_y, train_pred)
                    test_acc = accuracy_score(test_y, test_pred)

                    report = classification_report(test_y, test_pred, output_dict=True)

                    metrics = {
                        "train_accuracy": float(train_acc),
                        "test_accuracy":  float(test_acc),
                        "overfitting_score": float(train_acc - test_acc),
                        "precision_macro": float(report["macro avg"]["precision"]),
                        "recall_macro":    float(report["macro avg"]["recall"]),
                        "f1_macro":        float(report["macro avg"]["f1-score"]),
                    }
                    model_scores[name] = metrics

                    logger.info(f"{name} Test {test_acc:.4f} | F1 {metrics['f1_macro']:.4f}")

                    # Best Models
                    if test_acc > best_score:
                        best_score, best_model, best_name = test_acc, model, name
                
                except Exception as e:
                    logger.error(f"Error While training {name} : {str(e)}")
                    continue
            
            if best_model is None:
                raise CSPException("No Model Trained Successfully.", sys)
            
            os.makedirs(self.config.root_dir, exist_ok=True)

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            save_bin(best_model, Path(model_path))

            # Detailed Final Report
            final_pred = best_model.predict(test_x)
            detailed_report = classification_report(test_y, final_pred, output_dict=True)

            results = {
                "best_model": best_name,
                "best_test_score": float(best_score),
                "model_comparison": model_scores,
                "detailed_classification_report": detailed_report,
                "confusion_matrix": confusion_matrix(test_y, final_pred).tolist(),
                "class_distribution": {
                    "train": train_y.value_counts().to_dict(),
                    "test":  test_y.value_counts().to_dict(),
                },
            }

            results_path = os.path.join(self.config.root_dir, "training_results.json")
            save_json(results, Path(results_path))

            logger.info("Training completed successfully")
            logger.info(f"Best model : {best_name}")
            logger.info(f"Best score : {best_score:.4f}")
            logger.info(f"Model saved to        : {model_path}")
            logger.info(f"Metrics JSON saved to : {results_path}")
        
        except Exception as e:
            logger.error(f"Training Pipeline Failed: {str(e)}")
            raise CSPException(e, sys)