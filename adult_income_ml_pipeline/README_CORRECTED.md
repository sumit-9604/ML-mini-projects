# Adult Income ML Pipeline 💼

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

A complete **end-to-end Machine Learning pipeline** that predicts whether a person earns more than **$50K/year** using the Adult Income dataset. This project demonstrates professional ML practices including data preprocessing, feature engineering, model training, evaluation, and modular code structure.

---

## 🎯 Project Overview

### Problem Statement
Predict whether a person's annual income exceeds **$50K** based on demographic and employment-related features.

**Task Type:** Binary Classification  
**Target Variable:** `Income` (<=50K or >50K)  
**Samples:** 32,561 records  
**Features:** 14 attributes  

### Objectives
✅ Build a production-ready ML pipeline  
✅ Compare multiple classification models  
✅ Handle categorical and numerical features  
✅ Achieve high model performance  
✅ Maintain clean, modular code structure  

---

## 📊 Dataset

### Source
[Adult Income Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)

### Features (14 total)

| Type | Features |
|------|----------|
| **Numerical** | Age, Capital Gain, Capital Loss, Hours per Week |
| **Categorical** | Workclass, Education, Marital Status, Occupation, Relationship, Race, Sex, Native Country |
| **Target** | Income (Binary: <=50K, >50K) |

### Data Statistics
```
Total Samples:        32,561
Training Samples:     24,421
Test Samples:         8,140
Missing Values:       Yes (handled in preprocessing)
Class Distribution:   Imbalanced (~24% >50K, ~76% <=50K)
```

---

## 🏗️ Project Architecture

### Pipeline Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. DATA LOADING                                         │
│    • Load dataset                                       │
│    • Explore structure                                  │
│    • Check for missing values                           │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ 2. DATA PREPROCESSING                                   │
│    • Handle missing values (mean/mode imputation)       │
│    • Label encode categorical features                  │
│    • One-hot encode multi-class features               │
│    • Scale numerical features (StandardScaler)          │
│    • Train-test split (80/20)                           │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING                                       │
│    • Logistic Regression                                │
│    • Decision Tree Classifier                           │
│    • Random Forest Classifier                           │
│    • Gradient Boosting Classifier                       │
│    • Support Vector Machine                             │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ 4. MODEL EVALUATION                                     │
│    • Accuracy, Precision, Recall, F1 Score             │
│    • Confusion Matrix                                   │
│    • ROC Curve & AUC Score                              │
│    • Feature Importance Analysis                        │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ 5. PREDICTION & RESULTS                                 │
│    • Generate predictions                               │
│    • Compare model performance                          │
│    • Select best model                                  │
│    • Save trained model                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies & Libraries

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data Processing** | NumPy, Pandas |
| **ML Framework** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter Notebook |
| **Version Control** | Git |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step-by-Step Setup

**1. Clone the Repository**
```bash
git clone https://github.com/sumit-9604/ML-mini-projects.git
cd adult_income_ml_pipeline
```

**2. Create Virtual Environment**
```bash
python -m venv venv
```

**3. Activate Virtual Environment**

On **Windows**:
```bash
venv\Scripts\activate
```

On **macOS/Linux**:
```bash
source venv/bin/activate
```

**4. Install Dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 🚀 Running the Project

### Option 1: Using Jupyter Notebook (Recommended)

```bash
# Start Jupyter
jupyter notebook

# Open the notebook
# adult_income_pipeline.ipynb

# Run cells in order
```

### Option 2: Using Python Script

```bash
python src/setup.py
```

### Option 3: Using Pipeline Module

```python
from src.pipeline import AdultIncomePipeline

# Initialize pipeline
pipeline = AdultIncomePipeline()

# Load and preprocess data
pipeline.load_data('data/adult.csv')
pipeline.preprocess()

# Train models
pipeline.train_models()

# Evaluate and compare
results = pipeline.evaluate()

# Get predictions
predictions = pipeline.predict(X_test)
```

---

## 📈 Results & Performance

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC Score |
|-------|----------|-----------|--------|----------|-----------|
| **Logistic Regression** | 82.4% | 0.63 | 0.48 | 0.55 | 0.89 |
| **Decision Tree** | 81.2% | 0.60 | 0.45 | 0.52 | 0.77 |
| **Random Forest** | **84.6%** | **0.68** | **0.52** | **0.59** | **0.91** |
| **Gradient Boosting** | 84.2% | 0.67 | 0.51 | 0.58 | 0.90 |
| **SVM** | 79.8% | 0.58 | 0.42 | 0.49 | 0.87 |

### 🏆 Best Model: Random Forest
- **Accuracy:** 84.6%
- **Precision:** 0.68 (68% of predicted >50K are correct)
- **Recall:** 0.52 (catches 52% of actual >50K earners)
- **F1 Score:** 0.59
- **AUC:** 0.91

### Key Insights

📌 **Top 5 Important Features:**
1. **Age** - Strongest predictor of high income
2. **Hours per Week** - More hours correlate with higher income
3. **Education** - Higher education level increases income probability
4. **Occupation** - Certain occupations have higher average income
5. **Marital Status** - Married individuals tend to earn more

📌 **Model Performance:**
- Random Forest achieves best balance of precision and recall
- Training time: < 1 minute for full dataset
- Prediction accuracy: 84.6% on test set

---

## 📁 Project Structure

```
adult_income_ml_pipeline/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
│
├── data/
│   ├── adult.csv                      # Raw dataset
│   └── adult_processed.csv            # Processed data
│
├── notebooks/
│   └── adult_income_pipeline.ipynb    # Main notebook
│
├── src/
│   ├── __init__.py
│   ├── setup.py                       # Pipeline setup
│   ├── data_loader.py                 # Load data
│   ├── preprocessor.py                # Data preprocessing
│   ├── model_trainer.py               # Model training
│   ├── evaluator.py                   # Model evaluation
│   └── pipeline.py                    # Main pipeline class
│
├── models/
│   └── best_model.pkl                 # Trained model (saved)
│
└── results/
    ├── model_comparison.csv           # Performance metrics
    ├── predictions.csv                # Model predictions
    └── plots/
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── feature_importance.png
        └── model_comparison.png
```

---

## 💡 Key Features

✨ **Production-Ready Pipeline**
- Modular, reusable code structure
- Easy to extend with new models
- Clean separation of concerns

✨ **Comprehensive Data Handling**
- Missing value imputation
- Feature scaling and normalization
- Categorical variable encoding

✨ **Multiple Models**
- 5 different classification algorithms
- Automatic model comparison
- Best model selection

✨ **Detailed Evaluation**
- Multiple performance metrics
- Confusion matrix analysis
- Feature importance ranking
- ROC curve visualization

✨ **Easy to Use**
- Simple API for training and prediction
- Jupyter notebook for exploration
- Command-line interface available

---

## 🔧 How to Extend the Pipeline

### Add a New Model

```python
# In src/model_trainer.py

from sklearn.ensemble import AdaBoostClassifier

def add_adaboost_model(self):
    model = AdaBoostClassifier(n_estimators=100, random_state=42)
    self.models['AdaBoost'] = model
    return model
```

### Add New Preprocessing Step

```python
# In src/preprocessor.py

def add_polynomial_features(self, degree=2):
    """Add polynomial features for numerical columns"""
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    self.X_processed = poly.fit_transform(self.X_processed)
    return self.X_processed
```

### Save and Load Trained Model

```python
import pickle

# Save model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Load model
with open('models/best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Make predictions
predictions = loaded_model.predict(X_new)
```

---

## 🚀 Next Steps & Improvements

### Implemented ✅
- [x] Basic ML pipeline
- [x] Multiple model training
- [x] Model evaluation & comparison
- [x] Feature importance analysis

### In Progress 🔄
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Cross-validation implementation
- [ ] Class imbalance handling (SMOTE)

### Future Enhancements 📋
- [ ] Model deployment (FastAPI/Flask)
- [ ] REST API development
- [ ] Docker containerization
- [ ] MLflow experiment tracking
- [ ] Automated model retraining
- [ ] Data pipeline automation
- [ ] Feature store implementation

---

## 📊 Usage Examples

### Example 1: Train and Evaluate Models

```python
from src.pipeline import AdultIncomePipeline

# Initialize pipeline
pipeline = AdultIncomePipeline()

# Load data
pipeline.load_data('data/adult.csv')

# Preprocess
pipeline.preprocess()

# Train all models
pipeline.train_models()

# Get evaluation results
results = pipeline.evaluate()
print(results)
```

### Example 2: Make Predictions

```python
# Using trained model
predictions = pipeline.predict(X_test)

# Get probability scores
probabilities = pipeline.predict_proba(X_test)

# Save predictions
pipeline.save_predictions('results/predictions.csv', predictions)
```

### Example 3: Feature Analysis

```python
# Get feature importance
importance = pipeline.get_feature_importance()

# Plot feature importance
pipeline.plot_feature_importance()
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Guidelines
- Follow PEP 8 code style
- Add docstrings to functions
- Write clear commit messages
- Test your changes

---

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 👤 Author

**Sumit**  
📧 Email: [your-email@example.com]  
🔗 GitHub: [@sumit-9604](https://github.com/sumit-9604)  
💼 LinkedIn: [Your LinkedIn Profile]  

### About
- 🎓 AI/ML Engineer (Beginner)
- 💻 Passionate about Machine Learning
- 🤖 Interested in AI Pipelines & Deep Learning
- 🚀 Building Full Stack AI Systems

---

## 📚 Resources & References

### Useful Links
- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [ML Best Practices](https://developers.google.com/machine-learning/guides)

### Papers & Articles
- [Classification Metrics Explained](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38248)
- [Feature Importance Analysis](https://christophm.github.io/interpretable-ml-book/)
- [Binary Classification Best Practices](https://towardsdatascience.com/20-machine-learning-recipes-for-classification-3686410b2e90)

---

## ❓ FAQ

**Q: How long does it take to run the entire pipeline?**  
A: Approximately 2-5 minutes depending on your hardware.

**Q: Can I use this pipeline with different datasets?**  
A: Yes! The pipeline is modular and can be adapted for other classification datasets.

**Q: How do I improve model performance?**  
A: Try hyperparameter tuning, feature engineering, or collecting more data.

**Q: Is the model production-ready?**  
A: Yes, the pipeline can be deployed with proper monitoring and maintenance strategies.

**Q: Can I use this for real-world predictions?**  
A: The model can provide estimates, but should be used alongside human judgment for important decisions.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Adult Dataset
- Scikit-learn for ML algorithms
- The open-source community for tools and libraries

---

## 📧 Contact & Support

**Have questions or suggestions?**

- 📝 Open an Issue on GitHub
- 💌 Send an email
- 💬 Start a Discussion

---

## 📈 Project Stats

![Stars](https://img.shields.io/github/stars/sumit-9604/ML-mini-projects?style=social)
![Forks](https://img.shields.io/github/forks/sumit-9604/ML-mini-projects?style=social)
![Watchers](https://img.shields.io/github/watchers/sumit-9604/ML-mini-projects?style=social)

---

**⭐ If you found this project helpful, please consider giving it a star!**

---

*Last Updated: April 2026*  
*Status: Active & Maintained* ✅
