# End-to-End-Customer-Satisfaction-Prediction

This is an end-to-end machine learning operations (MLOps) project for classifying customer support tickets based on their priority level. The project implements a complete ML pipeline with data ingestion, validation, transformation, model training, evaluation, and deployment.

## Project Overview

The system classifies customer support tickets into different priority levels (High, Medium, Low) based on:
- Ticket Type
- Ticket Subject  
- Ticket Description
- Product Purchased

## Project Structure

The project follows a modular structure with clear separation of concerns:

- **Components**: Core ML pipeline components (data ingestion, validation, transformation, training, evaluation)
- **Pipeline**: Stage-wise pipeline execution
- **Entity**: Configuration and artifact entities
- **Utils**: Utility functions
- **Config**: Configuration management
- **Templates**: Web interface templates

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training pipeline:
   ```bash
   python main.py
   ```
5. Start the web application:
   ```bash
   python app.py
   ```

## Features

- **Modular Design**: Clean, maintainable code structure
- **Configuration Management**: YAML-based configuration
- **Logging**: Comprehensive logging throughout the pipeline
- **Exception Handling**: Custom exception handling
- **Model Comparison**: Trains and compares multiple models
- **Web Interface**: Flask-based web application for predictions
- **Data Validation**: Automated data quality checks

## Models Used

- Random Forest Classifier
- Logistic Regression
- XGBoost Classifier

The system automatically selects the best performing model based on test accuracy.

## API Endpoints

- `/`: Home page with prediction form
- `/train`: Trigger model training
- `/predict`: Make predictions on new tickets

## Technologies Used

- **ML Libraries**: scikit-learn, XGBoost
- **Web Framework**: Flask
- **Data Processing**: Pandas, NumPy
- **Text Processing**: CountVectorizer/TfidfVectorizer
- **Configuration**: PyYAML, python-box
- **Serialization**: Joblib

## Future Enhancements

- Integration with MLflow for experiment tracking
- Docker containerization
- CI/CD pipeline integration
- Real-time model monitoring
- A/B testing framework
```

## How to Run the Project

1. **Setup Environment**:
   ```bash
   git clone https://github.com/Durgeshsingh12712/End-to-End-Customer-Satisfaction-Prediction
   cd End-to-End-Customer-Satisfaction-Prediction
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   ```bash
   python main.py
   ```

3. **Run the Web Application**:
   ```bash
   python app.py
   ```

4. **Access the Application**:
   Open your browser and navigate to `http://localhost:5000`

This complete MLOps project provides a production-ready solution for customer support ticket classification with proper error handling, logging, configuration management, and a user-friendly web interface.