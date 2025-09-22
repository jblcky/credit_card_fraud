# Credit Card Fraud Detection System

## Overview
This project implements a machine learning-based credit card fraud detection system using XGBoost and neural networks. It includes model training, evaluation, and a user-friendly Streamlit web interface for real-time fraud detection.

## Project Structure
```
credit_card_fraud/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── xgboost_fraud_detector.joblib
│
├── notebooks/
│   ├── xgb.ipynb        # XGBoost model development
│   ├── model.ipynb      # Neural Network model development
│   └── app.py           # Streamlit web application
│
└── requirements.txt
```

## Features
- **Data Processing**: Handles imbalanced dataset with appropriate scaling
- **Model Development**:
  - XGBoost implementation with optimized hyperparameters
  - Neural Network alternative implementation
  - Proper train-test split with stratification
- **Performance Metrics**:
  - ROC-AUC curve analysis
  - Precision-Recall curve evaluation
  - Confusion matrix visualization
  - F1 score optimization
- **Interactive Web Interface**:
  - Real-time prediction capability
  - User-friendly input interface
  - Visual risk assessment
  - Confidence score display

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd credit_card_fraud
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Model Training
1. Open and run the notebooks in the following order:
   - `notebooks/xgb.ipynb` for XGBoost model
   - `notebooks/model.ipynb` for Neural Network model

### Web Application
1. Run the Streamlit application:
```bash
cd notebooks
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

## Model Performance
- XGBoost Model:
  - ROC-AUC Score: [Your Score]
  - Precision-Recall AUC: [Your Score]
  - F1 Score: [Your Score]

## Technical Details

### Features Used
- V1-V28: PCA-transformed transaction features
- Amount: Transaction amount
- Class: Binary target (0: normal, 1: fraud)

### Model Architecture
- XGBoost Classifier with:
  - Optimized depth and learning rate
  - Class weight balancing
  - Early stopping implementation

### Threshold Optimization
- Multiple threshold evaluations for:
  - Maximum F1 score
  - High recall scenarios
  - Precision-focused detection

## Web Interface Features
- Real-time transaction analysis
- Interactive gauge visualization
- Risk level categorization:
  - Low risk (< 0.5)
  - Medium risk (0.5 - 0.8)
  - High risk (> 0.8)
- Confidence score display

## Requirements
- Python 3.8+
- Key libraries:
  - streamlit >= 1.24.0
  - pandas >= 1.5.0
  - numpy >= 1.23.0
  - scikit-learn >= 1.0.0
  - xgboost >= 1.7.0
  - plotly >= 5.13.0
  - tensorflow (for neural network model)

## Future Improvements
- [ ] Model retraining pipeline
- [ ] Additional feature engineering
- [ ] API endpoint implementation
- [ ] Model performance monitoring
- [ ] Batch prediction capability

## License
[Your chosen license]

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
jasonling23@yahoo.com
