import re, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.utils import load_bin
from customerSatisfactionPrediction.exceptions import CSPException

warnings.filterwarnings('ignore')

class PredictionPipeline:
    def __init__(self):
        try:
            self.model = load_bin(Path('artifacts/model_trainer/model.pkl'))
            self.vectorizer = load_bin(Path("artifacts/data_transformation/vectorizer.pkl"))
            self.label_encoder = load_bin(Path("artifacts/data_transformation/label_encoder.pkl"))

            # Load Feature Metadata to ensure consistent feature dimentions
            try:
                self.feature_info = load_bin(Path("artifacts/data_transformation/feature_names.pkl"))
            except FileNotFoundError:
                logger.warning(f"Feature Metadata not found. Using Default configuration.")
                self.feature_info = {
                    'categorical_features': [],
                    'n_categorical_features': 0,
                    'n_text_features': 5000
                }
            
            # Store Excepted Feature count for validation
            self.expected_features = (
                len(self.feature_info['categorical_features']) + self.feature_info['n_text_features']
            )

            logger.info(f"Prediction Pipeline Initialized Successfully. Expected Feature: {self.expected_features}")
            logger.info(f"Available Priority Classes: {list(self.label_encoder.classes_)}")

        except Exception as e:
            logger.error(f"Failed to initialize Prediction Pipeline: {str(e)}")
            raise CSPException(e, sys)
        
    
    def preprocess_text(self, text):
        """Enhanced Text Preprocessing Matching Training Pipeline"""
        if pd.isna(text):
            return ""
        text = str(text).lower()

        #Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Remove Special Charactor
        text = re.sub(r'[^\w\s\-\.\!\?]', ' ', text)
        
        # Handle multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra whitespace
        text = text.strip()

        return text
    
    def _validate_inputs(self, ticket_type, ticket_subject, ticket_description, product_purchased):
        """Validate Input Parameters"""
        errors = []

        if not ticket_type or pd.isna(ticket_type):
            errors.append("ticket_type cannot be empty")
        
        if not ticket_subject or pd.isna(ticket_subject):
            errors.append("ticket_subject cannot be empty")
        
        if not ticket_description or pd.isna(ticket_description):
            errors.append("ticket_description cannot be empty")
        
        if errors:
            raise ValueError(f"Input Validation Failed: {', '.join(errors)}")
        
    def _create_categorical_features(self, ticket_type):
        """Create and Validate Categorical Features"""
        try:
            # Create Categorical Features
            categorical_df = pd.DataFrame({'Ticket Type': [ticket_type]})
            categorical_features = pd.get_dummies(
                categorical_df,
                columns=['Ticket Type'],
                drop_first=True,
                dtype=int
            )

            # Ensure we have the right categorical feature dimensions
            expected_categorical_features = self.feature_info['categorical_features']

            # Add Missing Feature with Zero Values
            for feature in expected_categorical_features:
                if feature not in categorical_features.columns:
                    categorical_features[feature] = 0
            
            # Reorder and select only expected features
            if expected_categorical_features:
                categorical_features = categorical_features.reindex(
                    columns=expected_categorical_features,
                    fill_value=0
                )
            logger.info(f"Categorical Features shape: {categorical_features.shape}")
            return categorical_features
        
        except Exception as e:
            logger.error(f"Error creating categorical features : {str(e)}")
            raise CSPException(e, sys)
        
    
    def _create_text_features(self, combined_text):
        """Create and Validate Text Features"""
        try:
            text_features = self.vectorizer.transform([combined_text])

            # Convert to Dense array for consistancy
            if hasattr(text_features, 'toarray'):
                text_features = text_features.toarray()
            
            text_df = pd.DataFrame(text_features)

            # Validate Feature count
            expected_text_features = self.feature_info['n_text_features']
            if text_df.shape[1] != expected_text_features:
                logger.warning(
                    f"Text Feature Dimenstion Mismatch. Expected: {expected_text_features},"
                    f"Got: {text_df.shape[1]}"
                )
            
            logger.info(f"Text features shape: {text_df.shape}")
            return text_df
        
        except Exception as e:
            logger.error(f"Error creating text features: {str(e)}")
            raise CSPException(e, sys)
        
    def get_model_info(self):
        """Get Information about the loaded model"""
        try:
            return {
                'model_type': type(self.model).__name__,
                'vectorizer_type': type(self.vectorizer).__name__,
                'priority_classes': type(self.label_encoder.classes_),
                'n_categorical_features': len(self.feature_info['categorical_features']),
                'n_text_features': self.feature_info['n_text_features'],
                'total_features': self.expected_features
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None
    
    def predict(self, ticket_type, ticket_subject, ticket_description, product_purchased):
        try:
            self._validate_inputs(ticket_type, ticket_subject, ticket_description, product_purchased)

            logger.info(f"Processing Prediction for ticket types: {ticket_type}")

            # Create Combined Text Feature
            if product_purchased and "{product_purchased}" in ticket_description:
                description = ticket_description.replace("{product_purchased}", str(product_purchased))
            else:
                description = str(ticket_description)
            
            # Combine description and subject for richer text Features
            combined_text = f"{description} {ticket_subject}"
            combined_text = self.preprocess_text(combined_text)

            logger.info(f"Combined Text (First 100 Chars): {combined_text[:100]}...")

            # Create Features
            categorical_features = self._create_categorical_features(ticket_type)
            text_features = self._create_text_features(combined_text)

            # Combine Feature
            if categorical_features.shape[1] > 0:
                all_features = pd.concat([
                    categorical_features.reset_index(drop=True),
                    text_features.reset_index(drop=True)
                ], axis=1)
            else:
                all_features = text_features
            
            # Validate final feature count
            if all_features.shape[1] != self.expected_features:
                logger.warning(
                    f"Feature count mismatch. Expected: {self.expected_features},"
                    f"Got: {all_features.shape[1]}"
                )

            # Make Prediction
            feature_array = all_features.values
            prediction = self.model.predict(feature_array)
            prediction_proba = self.model.predict_proba(feature_array)

            # Decode Prediction
            priority = self.label_encoder.inverse_transform(prediction)[0]
            confidence = float(np.max(prediction_proba))

            # Get Probability Distribution
            classes = self.label_encoder.classes_
            prob_dict = {
                classes[i]: float(prediction_proba[0][i])
                for i in range(len(classes))
            }

            logger.info(f"Prediction Made: {priority} with confidence: {confidence:.4f}")
            logger.info(f"Probability Distribution {prob_dict}")

            # Check for Potential Issue and provide insights
            if confidence > 0.95:
                logger.info("High confidence Prediction - Model seems very certain")
            elif confidence > 0.6:
                logger.warning("Low confidence Prediction - Model is uncertain")
            elif 0.6 <= confidence <=0.7:
                logger.info("Moderate Confidence Prediction - Consider Manual Review for edge cases ")
            
            return {
                'predicted_priority': priority,
                'confidence': confidence,
                'probability_distribution': prob_dict,
                'feature_count': all_features.shape[1],
                'text_length': len(combined_text)
            }
        except Exception as e:
            logger.error(f"Prediction Failded: {str(e)}")
            raise CSPException(e, sys)