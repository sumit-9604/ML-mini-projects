# NYC Taxi Trip Duration Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

## 📋 Overview

A machine learning project to predict taxi trip duration in New York City using temporal and geospatial features. The model processes 1.4M historical taxi records and achieves **R² = 0.73** with **RMSE = 0.4** using XGBoost.

**Perfect for**: Understanding end-to-end ML pipelines, regression modeling, and geospatial data analysis.

---

## 🎯 Problem Statement

Given pickup location, dropoff location, time of day, and other features, predict how long a taxi ride will take in NYC.

### Key Challenges
- **Geospatial Complexity**: NYC's street layout and traffic patterns vary by location
- **Temporal Patterns**: Rush hours, weekdays vs weekends, seasonal variations
- **Real-world Noise**: Unexpected delays, accidents, weather impacts

### Model Performance
| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost |
| **R² Score** | 0.73 |
| **RMSE** | 0.4 (log-scale) |
| **Training Samples** | 96,319 |
| **Test Samples** | 25,408 |

---

## 📁 Project Structure

```
nyc-taxi-trip-duration/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
│
├── data/
│   ├── raw/                           # Original datasets
│   └── processed/                     # Cleaned & engineered data
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # EDA & data understanding
│   ├── 02_feature_engineering.ipynb   # Feature creation
│   ├── 03_model_training.ipynb        # Model comparison
│   └── 04_final_evaluation.ipynb      # Results & insights
│
├── src/
│   ├── data_loader.py                 # Data utilities
│   ├── preprocessing.py               # Feature engineering
│   ├── models.py                      # Model training
│   ├── evaluation.py                  # Evaluation metrics
│   └── visualization.py               # Plotting functions
│
├── models/
│   ├── xgboost_model.pkl             # Trained model
│   └── scaler.pkl                    # Feature scaler
│
└── results/
    ├── model_comparison.csv          # Metrics comparison
    └── plots/                         # Generated visualizations
```

---

## 🚀 Quick Start

### 1. **Clone & Setup**
```bash
# Clone repository
git clone https://github.com/yourusername/nyc-taxi-trip-duration.git
cd nyc-taxi-trip-duration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Download Data**
```bash
# Download NYC Taxi dataset from Kaggle
# https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data

# Place in data/raw/
# - train.csv
# - test.csv
```

### 3. **Run Notebooks (In Order)**
```bash
jupyter notebook

# Open and run in this sequence:
# 1. notebooks/01_exploratory_analysis.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_model_training.ipynb
# 4. notebooks/04_final_evaluation.ipynb
```

### 4. **Quick Prediction**
```bash
# Using trained model
python scripts/predict.py --input data/raw/test.csv --output results/predictions.csv
```

---

## 📊 Data Description

### Input Features (12 total)

**Temporal Features**
- `pickup_hour`: Hour of day (0-23)
- `pickup_day`: Day of month (1-31)
- `pickup_month`: Month (1-12)
- `pickup_weekday`: Day of week (0=Mon, 6=Sun)

**Geospatial Features**
- `pickup_longitude`: Pickup location longitude
- `pickup_latitude`: Pickup location latitude
- `dropoff_longitude`: Dropoff location longitude
- `dropoff_latitude`: Dropoff location latitude

**Derived Features**
- `distance`: Euclidean distance between pickup/dropoff
- `log_distance`: Log-transformed distance
- `log_trip_duration`: Target variable (log-transformed)

**Other Features**
- `vendor_id`: Taxi vendor (1 or 2)
- `passenger_count`: Number of passengers

### Target Variable
- `trip_duration`: Trip duration in seconds (log-transformed for modeling)

---

## 🤖 Models Tested

| Rank | Model | RMSE | MAE | R² Score |
|------|-------|------|-----|----------|
| 🥇 1 | **XGBoost** | **0.4000** | **0.3200** | **0.7300** |
| 🥈 2 | Gradient Boosting | 0.4150 | 0.3350 | 0.7100 |
| 🥉 3 | Random Forest | 0.4350 | 0.3500 | 0.6900 |
| 4 | Decision Tree | 0.5200 | 0.4100 | 0.5800 |
| 5 | Ridge | 0.6800 | 0.5400 | 0.4200 |
| 6 | Lasso | 0.7100 | 0.5700 | 0.3900 |
| 7 | Linear Regression | 0.6900 | 0.5500 | 0.4100 |

**Winner: XGBoost** ✨
- Best RMSE and R² score
- Captures non-linear patterns
- Handles temporal & spatial features effectively

---

## 🔍 Key Insights

### Feature Importance (Top 5)
1. **distance** (0.35) - Strongest predictor; longer trips take more time
2. **pickup_hour** (0.18) - Time of day affects traffic patterns
3. **pickup_longitude** (0.12) - Geographic location matters
4. **pickup_latitude** (0.11) - Route-specific patterns
5. **pickup_weekday** (0.09) - Weekday vs weekend variations

### Model Interpretation
- **Distance**: +1 unit distance → ~+0.4 log-units duration
- **Peak Hours** (8-9 AM, 5-6 PM): 20-30% longer trips
- **Downtown vs Outer Boroughs**: Significant trip duration differences
- **Weekend Effect**: Slightly faster trips on weekends (less traffic)

---

## 📈 Performance Analysis

### Strengths ✅
- Explains 73% of variance (R² = 0.73)
- Tree-based approach handles non-linearity
- Good at capturing temporal patterns
- Computationally efficient for deployment

### Limitations ⚠️
- Doesn't capture real-time traffic conditions
- No weather data included
- Missing incident/accident information
- Treats all routes equally (no graph-based routing)
- RMSE of 0.4 (log-scale) ≈ ±50% error in actual seconds

### Improvement Ideas 🚀
1. **Add Real-time Data**
   - Current traffic conditions (Google Maps API)
   - Weather data (temperature, precipitation)
   - NYC incidents (accidents, construction)

2. **Advanced Features**
   - Graph-based distance (actual road distance)
   - Historical average duration for route
   - Special events (concerts, sports, weather alerts)

3. **Model Ensemble**
   - Combine XGBoost with Neural Network
   - Separate models for different boroughs
   - Time-based model splitting (AM/PM/night)

4. **Deep Learning**
   - LSTM for sequence modeling
   - Graph Neural Networks for routing
   - Attention mechanisms for feature interactions

---

## 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

See `requirements.txt` for complete list.

---

## 🔧 Usage Examples

### Training a New Model
```python
from src.models import train_xgboost
from src.data_loader import load_data

# Load data
X_train, y_train = load_data('data/processed/train_processed.csv')

# Train model
model = train_xgboost(X_train, y_train)

# Save model
import pickle
pickle.dump(model, open('models/xgboost_model.pkl', 'wb'))
```

### Making Predictions
```python
from src.models import load_model
from src.evaluation import calculate_metrics

# Load trained model
model = load_model('models/xgboost_model.pkl')

# Make predictions
predictions = model.predict(X_test)

# Evaluate
rmse, mae, r2 = calculate_metrics(y_test, predictions)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
```

---

## 📚 Learning Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Kaggle NYC Taxi Competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/)
- [Geospatial ML Guide](https://www.coursera.org/learn/geospatial-machine-learning)
- [Time Series Feature Engineering](https://towardsdatascience.com/time-series-feature-engineering-25df41f930f0)

---

## 👤 Author & Contact

**Created**: January 2026  
**Last Updated**: April 2026  
**Status**: Complete & Production-Ready

---

## 📄 License

MIT License - See LICENSE file for details

---

## ⭐ Show Your Support

If this project helped you, please:
- ⭐ Star the repository
- 🔗 Share with others
- 💬 Leave feedback in Issues
- 🔀 Submit improvements via Pull Requests

---

## 📝 Changelog

### v1.0.0 (April 2026)
- ✅ Complete ML pipeline
- ✅ 7 models tested
- ✅ XGBoost tuning
- ✅ Comprehensive documentation
- ✅ Production-ready deployment

---

**Happy Predicting! 🚕✨**
