# Customer Churn Prediction & Retention Recommendation System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn and provides actionable retention recommendations based on the customer's characteristics.

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave allows companies to take preventive action and improve customer retention.

This project uses the **IBM Telco Customer Churn dataset** to:

* Analyze customer behavior and churn patterns
* Clean and preprocess customer data
* Train multiple Machine Learning classification models
* Compare model performance
* Tune hyperparameters using GridSearchCV
* Predict customer churn probability
* Explain factors influencing the prediction
* Provide suggested customer retention actions
* Deploy the model using Streamlit

## 🎯 Problem Statement

The objective is to build a Machine Learning system that predicts whether a customer is likely to churn based on factors such as:

* Tenure
* Contract type
* Monthly charges
* Internet service
* Technical support
* Online security
* Payment method
* Billing preferences
* Customer demographics

The system also provides suggested actions that a business can consider to reduce the customer's churn risk.

---

## 📊 Dataset

The project uses the **Telco Customer Churn dataset**, containing information about telecom customers and whether they discontinued their service.

The dataset contains customer demographic, service, billing, and contract information.

### Target Variable

**Churn**

* `0` → Customer stayed
* `1` → Customer churned

During data cleaning, blank values in `TotalCharges` were converted to numeric values and rows with missing values were removed.

---

## 🔍 Exploratory Data Analysis

Several analyses were performed to understand the relationship between customer characteristics and churn.

### Key observations

* Customers with shorter tenure showed higher churn rates.
* Month-to-month contract customers had substantially higher churn rates than customers with long-term contracts.
* Customers with higher monthly charges showed higher average churn.
* Fiber optic customers showed a higher churn rate than DSL customers in this dataset.
* Tenure and TotalCharges showed a strong positive correlation.

### Important correlation

The correlation between `tenure` and `TotalCharges` was approximately:

**0.826**

This indicates a strong positive relationship between the two variables.

> Correlation indicates association and does not necessarily imply causation.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Removed the `customerID` column because it is an identifier rather than a predictive feature.
2. Converted `TotalCharges` from object/string to numeric.
3. Removed rows with missing `TotalCharges`.
4. Separated features (`X`) and target (`y`).
5. Converted the target:

   * `No → 0`
   * `Yes → 1`
6. Identified categorical and numerical features.
7. Applied One-Hot Encoding to categorical variables.
8. Used `handle_unknown="ignore"` to safely handle unseen categories.
9. Split the dataset into training and testing sets using an 80/20 split.
10. Used stratification to maintain the churn class distribution.

After preprocessing:

```text
Original features: 19
Features after encoding: 45
Training samples: 5625
Testing samples: 1407
```

---

## 🤖 Machine Learning Models

Three classification algorithms were evaluated:

### 1. Logistic Regression

Used as the primary baseline and final model because of its:

* Strong overall performance
* Interpretability
* Probability-based predictions
* Suitability for binary classification

### 2. Decision Tree

A Decision Tree was initially allowed to grow freely and showed significant overfitting.

The model achieved:

```text
Training Accuracy ≈ 99.88%
Testing Accuracy ≈ 72.21%
```

Hyperparameter tuning was then performed using `max_depth` and other tree parameters to reduce overfitting.

### 3. Random Forest

Random Forest was also evaluated and hyperparameters were optimized using GridSearchCV.

---

## 🔧 Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used to tune the Decision Tree and Random Forest models.

### Best Random Forest Parameters

```text
n_estimators = 200
max_depth = 10
min_samples_split = 2
min_samples_leaf = 2
```

The models were evaluated using **F1 score** during hyperparameter tuning because churn prediction involves an imbalanced target and both precision and recall are important.

---

## 📈 Model Performance

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression | **80.24%** | **64.46%** |     57.22% |     60.62% | **83.64%** |
| Tuned Decision Tree |     78.96% |     60.21% | **61.50%** | **60.85%** |     82.96% |
| Tuned Random Forest |     79.53% |     64.14% |     52.14% |     57.52% |     83.11% |

### Final Model

**Logistic Regression** was selected as the final model because it achieved:

* Highest accuracy
* Highest precision
* Highest ROC-AUC
* Competitive F1 score
* Better interpretability and simpler deployment

However, the tuned Decision Tree achieved slightly higher recall, which may be preferable if the business prioritizes identifying as many potential churners as possible.

---

## 🧠 Explainable AI

The Streamlit application provides an explanation of the prediction using the coefficients of the trained Logistic Regression model.

The application identifies:

### 🔴 Factors increasing churn risk

Features with positive contributions toward the churn class are displayed as potential factors increasing predicted churn risk.

### 🟢 Factors reducing churn risk

Features with negative contributions are displayed as factors associated with lower predicted churn risk.

This provides users with more insight than simply displaying a prediction.

> The feature contributions represent model-level associations and should not be interpreted as proof that a particular factor directly causes churn.

---

## 💡 Retention Recommendations

The application goes one step further than prediction.

Based on customer characteristics, it generates suggested retention actions.

Examples include:

### Month-to-month contract

**Suggested action:**
Offer a discounted long-term contract to encourage customer retention.

### High monthly charges

**Suggested action:**
Consider offering a personalized discount or a more affordable plan.

### Short tenure

**Suggested action:**
Provide an early-customer loyalty benefit or special offer.

### No technical support

**Suggested action:**
Offer technical support or a discounted support package.

### No online security

**Suggested action:**
Offer Online Security as an additional service.

### Fiber optic service

**Suggested action:**
Review service quality and customer satisfaction.

These recommendations are generated using a rule-based layer built on top of the Machine Learning prediction.

---

## 🌐 Streamlit Application

The trained model and preprocessing pipeline are integrated into a Streamlit application.

The user enters customer information and receives:

```text
Customer Churn Prediction
        ↓
Churn Probability
        ↓
Prediction
        ↓
Important Risk Factors
        ↓
Retention Recommendations
```

The application therefore provides both **prediction and actionable business insights**.

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── model/
│   ├── logistic_regression.pkl
│   └── preprocessor.pkl
│
├── notebook/
│   └── 01_customer_churn_prediction.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/AnshikaGupta-cloud/customer-churn-repository.git
```

### 2. Navigate to the project

```bash
cd customer-churn-repository
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**
* **Git & GitHub**

---

## 🔮 Future Improvements

Possible future improvements include:

* Implementing SHAP for more advanced explainability
* Testing additional models such as XGBoost
* Adding a customer retention cost-benefit analysis
* Adding customer segmentation
* Connecting the application to a database
* Deploying the Streamlit application online
* Adding an automated retention campaign recommendation system
* Monitoring model performance over time

---

## 👩‍💻 Author

**Anshika Gupta**

Machine Learning | Python | Data Analytics | AI

GitHub:
https://github.com/AnshikaGupta-cloud
