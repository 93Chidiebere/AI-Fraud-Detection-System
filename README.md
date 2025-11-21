# AI-Powered Fraud Detection System for Nigerian Banks

> **Real-time transaction network analysis and money mule detection system built for Nigerian banking hackathon (Track 5: Combating Digital Fraud)**

## 🎯 Project Overview

This AI-powered fraud detection system identifies fraudulent transactions and money mule networks using advanced machine learning and graph neural network techniques. The system analyzes transaction patterns, account behaviors, and network relationships to detect suspicious activities with **98% precision** and **99% F1-score**.

### Key Features

- ✅ **Real-time Transaction Analysis** - Process transactions in <50ms
- 🔍 **Network-Based Detection** - Identifies money mule accounts using graph analysis
- 🎯 **High Accuracy** - 98% precision with low false positive rate
- 📊 **Interactive Dashboard** - Streamlit-based web interface
- 🔒 **Risk Scoring** - Granular risk assessment (0-100%)
- 📈 **Feature Engineering** - 15 custom features including network metrics


## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Precision | 0.98 |
| Recall | 1.00 |
| F1-Score | 0.99 |
| ROC-AUC | 0.9991 |
| Fraud Detection Rate | 99.6% |
| False Positive Rate | <2% |

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- Internet connection for package installation

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

2. **Create virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required model files**

Ensure these files are in the project root:
- `xgb_fraud_model.pkl` (trained XGBoost model)
- `scaler.pkl` (feature scaler)

### Running the Application

**Launch Streamlit App:**
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
fraud-detection-system/
│
├── app.py                          # Streamlit web application
├── FraudDetectionTask.ipynb       # Data exploration & feature engineering
├── Fraud_ML_Model.ipynb           # Model training & evaluation
├── fraud.csv                       # Raw transaction data (not included - see Data)
├── fraud_features_ready.csv       # Processed features dataset
│
├── models/
│   ├── xgb_fraud_model.pkl        # Trained XGBoost model
│   └── scaler.pkl                 # StandardScaler for features
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore file
```

## 🔬 Technical Architecture

### Data Pipeline

```
Raw Transaction Data
        ↓
Feature Engineering (15 features)
        ↓
Network Graph Analysis (NetworkX)
        ↓
Feature Scaling (StandardScaler)
        ↓
XGBoost Classification
        ↓
Risk Score & Prediction
```

### Feature Engineering (15 Features)

#### 1. **Temporal Features**
- `step`: Transaction timestamp

#### 2. **Transaction Amount Features**
- `amount`: Transaction value
- `amount_to_origin_balance`: Amount as % of origin balance
- `amount_to_destination_balance`: Amount as % of destination balance

#### 3. **Balance Change Features**
- `origin_balance_change`: Change in sender balance
- `destination_balance_change`: Change in receiver balance

#### 4. **Error Detection Features**
- `origin_error`: Balance calculation mismatch (sender)
- `destination_error`: Balance calculation mismatch (receiver)

#### 5. **Zero Balance Flags**
- `origin_zero_after`: Sender account emptied (mule indicator)
- `destination_zero_before`: Receiver is new account (mule indicator)

#### 6. **Network Graph Features** (Key Innovation)
- `origin_out_degree`: Number of accounts sender transfers to
- `destination_in_degree`: Number of accounts transferring to receiver
- `origin_pagerank`: Sender's network importance score
- `destination_pagerank`: Receiver's network importance score
- `velocity`: Transaction frequency metric

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Filter to TRANSFER & CASH_OUT transactions (only fraud types)
   - Handle missing values (fillna with 0)
   - Feature scaling with StandardScaler

2. **Class Imbalance Handling**
   - Original fraud rate: 0.296%
   - SMOTE oversampling to 23.1%
   - Prevents model bias toward majority class

3. **Model Training**
   - **Base Model**: XGBoost Classifier
   - **Hyperparameter Tuning**: RandomizedSearchCV (20 iterations, 3-fold CV)
   - **Optimization Metric**: F1-macro (balanced scoring)

4. **Best Model Configuration**
```python
XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.2,
    gamma=0.1,
    subsample=0.7,
    colsample_bytree=1.0,
    min_child_weight=3
)
```

## 🎨 Streamlit Dashboard Features

### 1. Transaction Input Form
- Transaction type (TRANSFER/CASH_OUT)
- Amount (₦)
- Origin account balance
- Destination account balance
- Network connections count

### 2. Real-time Analysis
- **Risk Score Gauge** (0-100%)
- **Color-coded Risk Levels**:
  - 🟢 Green (0-30%): Low Risk - Approve
  - 🟡 Yellow (30-70%): Medium Risk - Manual Review
  - 🔴 Red (70-100%): High Risk - Block

### 3. Risk Factor Breakdown
- Origin account emptied warning
- New destination account flag
- Large transaction alert
- Network anomaly detection
- Transaction summary

### 4. Footer Metrics
- Model type (XGBoost)
- Precision score
- F1-score

## 📊 Dataset Information

### Source
- **Dataset**: PaySim Mobile Money Simulator
- **Format**: CSV (470MB uncompressed)
- **Records**: 6.36M transactions
- **Fraud Cases**: 8,213 (0.129%)
- **Time Period**: 1 month simulation
- **Geographic Focus**: African mobile money patterns

### Download Dataset
```bash
# Kaggle CLI (if you have it configured)
kaggle datasets download -d ealaxi/paysim1

# Or download manually from:
# https://www.kaggle.com/datasets/ealaxi/paysim1
```

### Data Fields
| Field | Description | Type |
|-------|-------------|------|
| step | Hour of transaction (1-744) | int |
| type | Transaction type | categorical |
| amount | Transaction amount | float |
| nameOrig | Sender account ID | string |
| oldbalanceOrg | Sender balance before | float |
| newbalanceOrig | Sender balance after | float |
| nameDest | Receiver account ID | string |
| oldbalanceDest | Receiver balance before | float |
| newbalanceDest | Receiver balance after | float |
| isFraud | Fraud label (target) | binary |
| isFlaggedFraud | System flag | binary |

## 🔧 Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Cloud Deployment Options

#### 1. Streamlit Cloud (Recommended)
```bash
# 1. Push to GitHub
git add .
git commit -m "Deploy fraud detection app"
git push origin main

# 2. Connect repository to Streamlit Cloud
# Visit: https://streamlit.io/cloud
# Click "New app" → Select repository
```

#### 2. Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT" > Procfile

# Deploy
heroku create fraud-detection-app
git push heroku main
```

#### 3. Docker
```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
# Build and run
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection
```

## 🧪 Testing

### Demo Scenarios

**1. High-Risk Fraud Transaction**
```
Type: TRANSFER
Amount: ₦500,000
Origin Balance: ₦500,000 (will be emptied)
Destination Balance: ₦0 (new account)
Connections: 1
Expected: 95%+ fraud probability
```

**2. Legitimate Large Transaction**
```
Type: TRANSFER
Amount: ₦200,000
Origin Balance: ₦5,000,000
Destination Balance: ₦100,000
Connections: 50
Expected: <10% fraud probability
```

**3. Borderline Case**
```
Type: CASH_OUT
Amount: ₦100,000
Origin Balance: ₦150,000
Destination Balance: ₦10,000
Connections: 3
Expected: 30-50% (manual review)
```

## 📈 Model Training

To retrain the model with new data:

1. **Prepare new transaction data**
```python
# Ensure CSV has required columns
df = pd.read_csv('new_fraud_data.csv')
```

2. **Run feature engineering**
```bash
jupyter notebook FraudDetectionTask.ipynb
# Execute all cells
```

3. **Train and save model**
```bash
jupyter notebook Fraud_ML_Model.ipynb
# Model saved as xgb_fraud_model.pkl
```

4. **Evaluate performance**
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request


## 👤 Author

**Vincent Chidiebere**
- GitHub: (https://github.com/93Chidiebere)
- LinkedIn: (https://www.linkedin.com/in/chidiebere-christopher/)

## 🙏 Acknowledgments

- PaySim dataset creators (EALAXI)
- Nigerian Banking Hackathon organizers
- XGBoost development team
- Streamlit community

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: vchidiebere.vc@gmail.com

## 🎓 Research & References

1. **PaySim Paper**: [Synthetic Financial Datasets for Fraud Detection](https://arxiv.org/abs/1601.05547)
2. **XGBoost**: Chen & Guestrin (2016) - [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
3. **SMOTE**: Chawla et al. (2002) - Synthetic Minority Over-sampling Technique
4. **Graph-based Fraud Detection**: Network analysis in financial crime detection

## 🔮 Future Enhancements

- [ ] Real-time streaming data support (Apache Kafka)
- [ ] Deep learning models (LSTM for temporal patterns)
- [ ] Explainable AI (SHAP values visualization)
- [ ] Multi-currency support
- [ ] API integration for banks
- [ ] Automated retraining pipeline
- [ ] Mobile app version

---

**Built with ❤️ for Nigerian Banking Innovation**

*Last Updated: November 2025*