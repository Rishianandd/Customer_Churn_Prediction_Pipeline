# 📊 Customer Churn Prediction — End-to-End ML Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=flat&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-green?style=flat" />
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-orange?style=flat" />
</p>

---

## 🚀 Project Overview

This project implements an **end-to-end Customer Churn Prediction pipeline** using **Python, Pandas, and Scikit-learn**.

The objective is to predict whether a customer is likely to **churn (leave a service)** based on demographic, service, account, and billing-related information.

The project demonstrates a complete machine learning workflow, starting from raw customer data and progressing through data cleaning, preprocessing, feature transformation, model training, prediction, and evaluation.

### 🔄 Overall Pipeline

```text
Raw Customer Data
        ↓
Data Inspection
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Target Encoding
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Numerical & Categorical Preprocessing
        ↓
StandardScaler + OneHotEncoder
        ↓
Logistic Regression
        ↓
Predictions
        ↓
Model Evaluation
        ↓
Accuracy / Precision / Recall / F1
        ↓
Confusion Matrix
```

---

## 🎯 Problem Statement

Customer churn is an important business problem because losing existing customers can negatively impact revenue and long-term business growth.

The goal of this project is to build a machine learning classification model that can identify customers who are likely to leave a service.

The model predicts two classes:

| Value | Meaning                 |
| ----- | ----------------------- |
| `0`   | Customer will not churn |
| `1`   | Customer will churn     |

The model can therefore be used as a baseline predictive system for identifying customers who may require additional retention strategies.

---

## 📂 Dataset

The project uses a **customer churn dataset** containing demographic, service, account, and billing information.

The dataset contains features representing different aspects of customer behavior and service usage.

### Dataset Structure

```text
customer-churn-prediction/
│
├── customer_churn/
│   └── customer_churn.csv
│
├── pipeline.ipynb
│
└── README.md
```

### Target Variable

```text
Churn
```

Original values:

```text
Yes
No
```

Encoded values:

```text
Yes → 1
No  → 0
```

---

# 🔎 Exploratory Data Analysis

Before training the model, the dataset was inspected to understand its structure and identify potential data quality issues.

### Initial Inspection

```python
df.head()
```

Used to inspect the first few records.

```python
df.info()
```

Used to examine:

* Number of records
* Feature names
* Data types
* Non-null values

```python
df.isnull().sum()
```

Used to identify missing values across the dataset.

---

# 🧹 Data Preprocessing

Data preprocessing is an important stage of the machine learning pipeline because raw datasets often contain unnecessary identifiers, categorical variables, duplicates, and features that require transformation.

## 1️⃣ Data Inspection

The dataset was first examined for:

* Missing values
* Data types
* Duplicate records
* Feature distributions
* Target variable structure

---

## 2️⃣ Duplicate Removal

Duplicate records were identified and removed to prevent repeated observations from influencing the model.

```python
df = df.drop_duplicates()
```

This helps ensure that each customer record contributes appropriately to the training process.

---

## 3️⃣ Target Encoding

The target column `Churn` contained categorical values:

```text
No
Yes
```

These values were converted into binary numerical labels:

```text
No  → 0
Yes → 1
```

This transformation allows Logistic Regression to perform binary classification.

---

## 4️⃣ Removing Customer ID

The `customerID` column was removed because it is an identifier rather than a meaningful predictive feature.

```text
customerID
```

was therefore excluded from the feature set.

---

# 🔢 Feature Preparation

The dataset contains both **numerical** and **categorical** variables.

These feature types require different preprocessing techniques.

---

## 🔢 Numerical Features

Numerical features were standardized using:

```python
StandardScaler()
```

Standardization transforms numerical variables to a comparable scale.

This is particularly useful for Logistic Regression because the model can be affected by differences in feature scales.

---

## 🔤 Categorical Features

Categorical variables were transformed using:

```python
OneHotEncoder(handle_unknown='ignore')
```

One-hot encoding converts categorical values into numerical representations that can be processed by the machine learning model.

The `handle_unknown='ignore'` option also prevents errors when previously unseen categories appear during transformation.

---

# 🔀 Train-Test Split

The dataset was divided into training and testing sets using an **80/20 split**.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Configuration

| Parameter      | Value   |
| -------------- | ------- |
| Training Data  | 80%     |
| Testing Data   | 20%     |
| Random State   | 42      |
| Stratification | Enabled |

### Why Stratification?

`stratify=y` maintains a similar distribution of the target classes in both the training and testing datasets.

---

# 🔗 Preprocessing Pipeline

A Scikit-learn `ColumnTransformer` was used to apply different transformations to numerical and categorical features.

The preprocessing workflow can be represented as:

```text
                    Input Features
                          │
              ┌───────────┴───────────┐
              │                       │
        Numerical Features      Categorical Features
              │                       │
              ▼                       ▼
       StandardScaler          OneHotEncoder
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                 Processed Features
```

---

# 🤖 Machine Learning Model

## Logistic Regression

The primary model used in this project is **Logistic Regression**.

Logistic Regression was selected because the objective is a **binary classification problem**.

The model estimates the probability of a customer belonging to a particular class.

```text
0 → No Churn
1 → Churn
```

---

# 🧩 Complete ML Pipeline

The preprocessing and classification model were combined into a single Scikit-learn pipeline.

```python
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])
```

This provides several advantages:

* ✅ Consistent preprocessing
* ✅ Reproducible workflow
* ✅ Prevents preprocessing inconsistencies
* ✅ Simplifies training and prediction
* ✅ Keeps transformations and model together

---

# 🏋️ Model Training

The model was trained using the training dataset:

```python
model_pipeline.fit(X_train, y_train)
```

The pipeline automatically:

1. Identifies numerical and categorical features.
2. Applies the appropriate transformations.
3. Generates the processed feature matrix.
4. Trains the Logistic Regression classifier.

---

# 🔮 Prediction

Predictions were generated on the unseen test dataset:

```python
y_pred = model_pipeline.predict(X_test)
```

The trained model therefore predicts churn behavior for customers that were not used during training.

---

# 📊 Model Evaluation

The model was evaluated using multiple classification metrics.

### Metrics Used

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

---

## 📈 Results

| Metric        |      Score |
| ------------- | ---------: |
| **Accuracy**  | **79.42%** |
| **Precision** | **63.13%** |
| **Recall**    | **54.01%** |
| **F1 Score**  | **58.21%** |

---

## 📌 Metric Interpretation

### Accuracy — 79.42%

Approximately **79.42% of the test observations were classified correctly**.

---

### Precision — 63.13%

Precision measures how many of the customers predicted as churners actually belonged to the churn class.

A precision of **63.13%** indicates that approximately 63% of positive churn predictions were correct.

---

### Recall — 54.01%

Recall measures how many of the actual churn cases were successfully identified by the model.

A recall of **54.01%** means the model identified approximately 54% of the actual churn cases.

---

### F1 Score — 58.21%

The F1 score combines precision and recall into a single metric.

The resulting F1 score is:

```text
58.21%
```

This provides a balanced view of the model's ability to identify churn cases.

---

# 📉 Confusion Matrix

A confusion matrix was used to analyze the model's classification performance.

It provides four categories:

| Category       | Description                                |
| -------------- | ------------------------------------------ |
| True Positive  | Correctly predicted churn                  |
| True Negative  | Correctly predicted non-churn              |
| False Positive | Predicted churn but customer did not churn |
| False Negative | Actual churn but predicted non-churn       |

For churn prediction, **False Negatives** are particularly important because they represent customers who actually churned but were not identified by the model.

---

# 📊 Visualization

The project includes data visualizations to support the analysis and interpretation of model performance.

Visualizations include:

* 📊 Feature analysis
* 📈 Distribution plots
* 📉 Classification performance
* 🔲 Confusion matrix

These visualizations help understand both the underlying dataset and the model's predictions.

---

# 🧠 Key Learning Outcomes

This project demonstrates practical knowledge of the complete machine learning workflow.

### Data Handling

* Data loading
* Dataset inspection
* Missing value analysis
* Duplicate removal
* Feature selection

### Data Preprocessing

* Target encoding
* Numerical feature scaling
* Categorical feature encoding
* ColumnTransformer

### Machine Learning

* Train-test splitting
* Logistic Regression
* Scikit-learn Pipeline
* Model training
* Prediction

### Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

---

# 🔄 End-to-End Architecture

```text
                 CUSTOMER DATASET
                       │
                       ▼
                DATA INSPECTION
                       │
                       ▼
                 DATA CLEANING
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Remove Duplicates    Remove Customer ID
             │                   │
             └─────────┬─────────┘
                       ▼
                 TARGET ENCODING
                       │
                       ▼
                 FEATURE / TARGET
                   SEPARATION
                       │
                       ▼
                 TRAIN / TEST SPLIT
                       │
                       ▼
              COLUMN TRANSFORMER
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      StandardScaler       OneHotEncoder
             │                   │
             └─────────┬─────────┘
                       ▼
              LOGISTIC REGRESSION
                       │
                       ▼
                  PREDICTIONS
                       │
                       ▼
              MODEL EVALUATION
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Accuracy    Precision     Recall
                       │
                       ▼
                    F1 Score
                       │
                       ▼
               CONFUSION MATRIX
```

---

# 📁 Project Structure

```text
customer-churn-prediction/
│
├── customer_churn/
│   └── customer_churn.csv
│
├── pipeline.ipynb
│
└── README.md
```

---

# 🚀 Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
```

Navigate into the project:

```bash
cd customer-churn-prediction
```

---

## 2️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 3️⃣ Launch Jupyter Notebook

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

---

## 4️⃣ Open the Notebook

```text
pipeline.ipynb
```

Run the notebook cells sequentially to reproduce the complete machine learning workflow.

---

# 💡 Possible Improvements

The current implementation provides a Logistic Regression baseline. Further experiments could investigate:

### ⚖️ Class Imbalance

* SMOTE
* Class weights
* Random oversampling
* Random undersampling

### 🎯 Hyperparameter Optimization

* GridSearchCV
* RandomizedSearchCV
* Cross-validation

### 🤖 Alternative Models

* Random Forest
* XGBoost
* Gradient Boosting
* Support Vector Machine

### 📊 Additional Metrics

* ROC-AUC
* Precision-Recall AUC
* ROC Curve
* Precision-Recall Curve

### 🔧 Threshold Optimization

The classification threshold could be tuned depending on whether the business objective prioritizes:

* Higher recall
* Higher precision
* Balanced performance

---

# 📌 Project Summary

| Category          | Details                         |
| ----------------- | ------------------------------- |
| **Project Type**  | Machine Learning Classification |
| **Task**          | Customer Churn Prediction       |
| **Model**         | Logistic Regression             |
| **Framework**     | Scikit-learn                    |
| **Preprocessing** | StandardScaler + OneHotEncoder  |
| **Validation**    | 80/20 Train-Test Split          |
| **Accuracy**      | 79.42%                          |
| **Precision**     | 63.13%                          |
| **Recall**        | 54.01%                          |
| **F1 Score**      | 58.21%                          |
| **Environment**   | Jupyter Notebook                |

---

# 👨‍💻 Author

**Rishi Anand**

🎓 B.Tech — Computer Science & Engineering (Artificial Intelligence & Machine Learning)

🇫🇷 Current MSc Data Science & Analytics — EPITA Paris

---

## 📬 Contact

📧 **Email:** [rishianandv@gmail.com](mailto:rishianandv@gmail.com)

💼 **LinkedIn:** linkedin.com/in/rishiii-anand

🐙 **GitHub:** github.com/RishiAnandd

---

## ⭐ Acknowledgement

This project was developed as an **Employee Machine Learning Assessment** demonstrating an end-to-end approach to customer churn prediction using Python and Scikit-learn.

---

⭐ **If you found this project useful, consider giving the repository a star!**

```
```
