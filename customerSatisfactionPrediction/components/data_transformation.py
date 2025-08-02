import os, re
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.exceptions import CSPException
from customerSatisfactionPrediction.utils import save_bin
from customerSatisfactionPrediction.entity import DataTransformationConfig, DataTransformationArtifact


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def get_data_transformer_object(self):
        try:
            if self.config.tokenizer_name == "CountVectorizer":
                vectorizer = CountVectorizer(
                    max_features=self.config.max_features,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
            else:
                vectorizer = TfidfVectorizer(
                    max_features=self.config.max_features,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
            return vectorizer
        except Exception as e:
            logger.error(f"Error Creating Vectorizer: {str(e)}")
            raise CSPException(e)
    
    def validate_data(self, df):
        """Validate Input Data Quality"""
        try:
            required_columns = ["Ticket Description", "Product Purchased", "Ticket Subject", "Ticket Type"]
            misssing_cols = [col for col in required_columns if col not in df.columns]

            if misssing_cols:
                raise ValueError(f"Missing Required Columns: {misssing_cols}")
            
            for col in required_columns:
                misssing_pct = df[col].isna().sum() / len(df)
                if misssing_pct > 0.5:
                    logger.warning(f"Column '{col}' has {misssing_pct:.2%} missing values")
                elif misssing_pct > 0:
                    logger.info(f"Column '{col}' has {misssing_pct:.2%} missing values")

            return True
        except Exception as e:
            logger.error(f"Data Validation Failed: {str(e)}")
            raise CSPException(e)
    
    def create_combined_text(self, row):
        """Create combined text feature with error handling"""
        try:
            description = str(row.get("Ticket Description", ""))
            product = str(row.get("Product Purchased", ""))
            subject = str(row.get("Ticket Subject", ""))

            if "{product_purchased}" in description:
                description = description.replace("{product_purchased}", product)
            
            combined = f"{description} {subject}"
            return self.preprocess_text(combined)
        except Exception as e:
            logger.warning(f"Error Processing Text for row: {e}")
            return ""
    
    def initiate_data_transformation(self):
        try:
            df = pd.read_csv(self.config.data_path)
            logger.info(f"Dataset Loaded with shape: {df.shape}")

            # Validate Data Quality
            self.validate_data(df)

            df = df.dropna().reset_index(drop=True)
            logger.info(f"After Drop None Value: {df.shape}")

            df = df.drop_duplicates().reset_index(drop=True)
            logger.info(f"After Removing Duplicates: {df.shape}")

            # Handle Target Column naming
            if 'Ticket Priority' in df.columns:
                df['Ticket_Priority'] = df['Ticket Priority']
                df = df.drop('Ticket Priority', axis=1)
                logger.info("Renamed  'Ticket Priority' to Ticket_Priority")
            elif 'Ticket_Priority' not in df.columns:
                raise ValueError("Target Column 'Ticket Priority' or 'Ticket_Priority' not Found in dataset")
            
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df['Ticket_Priority']
            )

            logger.info(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

            # Create Combined Text Feature
            train_df["combined_text"] = train_df.apply(self.create_combined_text, axis=1)
            test_df["combined_text"] = test_df.apply(self.create_combined_text, axis=1)

            categorical_features = ["Ticket Type"]

            # Validate Categorical Feature exist
            available_categorical = [col for col in categorical_features if col in train_df.columns]
            if not available_categorical:
                logger.warning("No categorical features found in dataset")
                # Create empty categorical DataFrames
                train_categorical = pd.DataFrame(index=train_df.index)
                test_categorical = pd.DataFrame(index=test_df.index)
            else:
                # Create Dummy Variables
                train_categorical = pd.get_dummies(
                    train_df[available_categorical],
                    columns=available_categorical,
                    drop_first=True,
                    dtype=int
                )
                test_categorical = pd.get_dummies(
                    test_df[available_categorical],
                    columns=available_categorical,
                    drop_first=True,
                    dtype=int
                )

                # Ensure test has same columns as train
                for col in train_categorical.columns:
                    if col not in test_categorical.columns:
                        test_categorical[col] = 0
                
                # Reorder Columns to Match
                test_categorical = test_categorical[train_categorical.columns]
            
            # Transform Text Features
            logger.info("Transforming Text Features...")
            preprocessing_obj = self.get_data_transformer_object()
            train_text_features = preprocessing_obj.fit_transform(train_df["combined_text"])
            test_text_features = preprocessing_obj.transform(test_df["combined_text"])

            # Log Feature dimenstions
            logger.info(f"Feature Dimenstion - Categorical: {train_categorical.shape[1] if not train_categorical.empty else 0}, Text: {train_text_features.shape[1]}")

            # Memory-efficient feature combination using sparse metrices
            if not train_categorical.empty:
                train_categorical_sparse = csr_matrix(train_categorical.values)
                test_categorical_sparse = csr_matrix(test_categorical.values)

                # Combine sparse metrices
                train_features_sparse = hstack([train_categorical_sparse, train_text_features])
                test_features_sparse = hstack([test_categorical_sparse, test_text_features])
            else:
                train_features_sparse = train_text_features
                test_features_sparse = test_text_features

            # Encode target variable
            label_encoder = LabelEncoder()
            train_target = label_encoder.fit_transform(train_df['Ticket_Priority'])
            test_target = label_encoder.transform(test_df['Ticket_Priority'])

            logger.info(f"Label Classes: {label_encoder.classes_}")

            # Creete Final Arrays
            train_arr = np.c_[train_features_sparse.toarray(), train_target]
            test_arr = np.c_[test_features_sparse.toarray(), test_target]

            logger.info(f"Final Feature shape - train: {train_arr.shape}, Test: {test_arr.shape}")

            # Create output directory if it doesm't exist
            os.makedirs(self.config.root_dir, exist_ok=True)

            # Save Preprocessing Object and data
            vectorizer_path = os.path.join(self.config.root_dir, "vectorizer.pkl")
            label_encoder_path = os.path.join(self.config.root_dir, "label_encoder.pkl")
            feature_names_path = os.path.join(self.config.root_dir, "feature_names.pkl")
            train_file_path = os.path.join(self.config.root_dir, "train.csv")
            test_file_path = os.path.join(self.config.root_dir, "test.csv")

            # Save object
            save_bin(preprocessing_obj, Path(vectorizer_path))
            save_bin(label_encoder, Path(label_encoder_path))
            save_bin(
                {
                    'categotical_features': list(train_categorical.columns) if not train_categorical.empty else [],
                    'n_categorical_features': train_categorical.shape[1] if not train_categorical.empty else 0,
                    'n_text_features': train_text_features.shape[1],
                    'label_classes': label_encoder.classes_.tolist()
                }, Path(feature_names_path)
            )

            # Save Data
            pd.DataFrame(train_arr).to_csv(train_file_path, index=False, header=False)
            pd.DataFrame(test_arr).to_csv(test_file_path, index=False, header=False)

            logger.info("Data Transformation Complated Successfully")
            logger.info(f"Files Saved to: {self.config.root_dir}")

            return DataTransformationArtifact(
                transformed_object_file_path=vectorizer_path,
                transformed_train_file_path=train_file_path,
                transformed_test_file_path=test_file_path
            )
        except Exception as e:
            logger.error(f"Data Transformation Failed: {str(e)}")
            raise e