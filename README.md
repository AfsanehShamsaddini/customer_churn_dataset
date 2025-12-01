
<div align="center">

# 📊🔥 Customer Churn Prediction  
### Machine Learning Project for Predicting Customer Loss

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![ML](https://img.shields.io/badge/ML-Supervised-orange?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Customer%20Churn-purple?style=for-the-badge)

</div>

---

# 📚 Table of Contents
- [📘 Project Overview](#-project-overview)
- [📁 Dataset](#-dataset)
- [🛠 Requirements](#-requirements)
- [⚙️ Installation](#️-installation)
- [🔍 Workflow](#-workflow)
- [📊 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🧹 Preprocessing](#-preprocessing)
- [🤖 Models Used](#-models-used)
- [📈 Results](#-results)
- [📬 Contact](#-contact)

---

# 📘 Project Overview

Customer churn is a major challenge for subscription-based businesses.  
This project aims to **predict customer churn** using machine learning and identify the **key behavioral factors** behind churn.

🎯 **Goal**: Build, analyze, and compare ML models to classify customers as *Churn* or *Not Churn*.

Key Features:
- Full EDA with visualizations  
- Data cleaning + preprocessing  
- Handling imbalanced data using **SMOTE**  
- Testing 7 machine learning models  
- Performance comparison using multiple metrics  

---

# 📁 Dataset

The dataset contains demographic and behavioral information:

| Column | Description |
|--------|-------------|
| **CustomerID** | Unique ID |
| **Gender** | Male / Female |
| **Age** | Customer age |
| **Subscription Type** | Basic / Standard / Premium |
| **Contract Length** | Monthly / Quarterly / Annual |
| **Total Spend** | Total amount paid |
| **Churn** | Target variable (0 = Active, 1 = Churned) |

📌 **Dataset Source:**  
`/content/drive/MyDrive/ML/churn/customer_churn_dataset-testing-master.csv`

---

# 🛠 Requirements

To install dependencies:

```bash
pip install -r requirements.txt
````

### Packages Used

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
```

---

# ⚙️ Installation

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

---

# 🔍 Workflow

### ✔️ Step 1 — Data Loading & Inspection

* Checking missing values
* Detecting mixed types
* Initial statistics

### ✔️ Step 2 — EDA (Exploratory Data Analysis)

* Distribution plots
* Churn vs categorical features
* Correlation heatmap

### ✔️ Step 3 — Data Preprocessing

* Encoding categorical variables
* Scaling numeric features
* Removing unnecessary columns
* Balancing using SMOTE

### ✔️ Step 4 — Model Training

Models used:

* Logistic Regression
* Random Forest
* KNN
* Decision Tree
* Support Vector Machine
* **XGBoost**
* Gradient Boosting

### ✔️ Step 5 — Evaluation & Visualization

Metrics:

* Accuracy
* Recall
* F1 Score
* ROC AUC

---

# 📊 Exploratory Data Analysis

Examples of visualizations included in the project:

* Age distribution by Gender
* Churn rate by Subscription Type
* Contract Length vs Churn
* Violin & Box plots
* Correlation heatmap

📌 *Plots are available inside the project notebook.*

---

# 🧹 Preprocessing

Includes:

* Mapping Gender → {0,1}
* One-hot encoding Subscription Type
* Ordinal encoding Contract Length
* Scaling with StandardScaler
* Handling imbalance using:
  ✔ SMOTE
  ✔ Balanced class weights

---

# 🤖 Models Used

| Model               | Notes                   |
| ------------------- | ----------------------- |
| Logistic Regression | Baseline model          |
| Random Forest       | Strong tree-based model |
| KNN                 | With SMOTE              |
| Decision Tree       | Fast & interpretable    |
| SVM                 | Balanced with SMOTE     |
| **XGBoost**         | Best performance        |
| Gradient Boosting   | Stable and robust       |

---

# 📈 Results

### 📊 1. Results Table

| Model                   | Accuracy | Recall Score | F1 Score | ROC AUC Score |
|-------------------------|----------|--------------|----------|---------------|
| Logistic Regression     | 0.831146 | 0.847254     | 0.825801 | 0.831989      |
| Random Forest           | 0.998369 | 0.997369     | 0.998272 | 0.998317      |
| K-Nearest Neighbors     | 0.906252 | 0.939658     | 0.904487 | 0.908001      |
| Decision Tree           | 0.998680 | 0.998520     | 0.998602 | 0.998671      |
| Support Vector Machine  | 0.941359 | 0.957415     | 0.939118 | 0.942199      |
| XGBoost                 |0.999845  | 0.999836     | 0.999836 | 0.999844      |
| Gradient Boosting       | 0.996194 | 0.994245     | 0.995965 | 0.996092      |


---

### 📉 2. Grouped Bar Chart (Model Comparison)

![Performance Bar Chart](https://github.com/AfsanehShamsaddini/customer_churn_dataset/blob/main/image/download.png?raw=true)


---

### 🏆 Best Performing Models

| Rank    | Model         | Highlight                                         |
| ------- | ------------- | ------------------------------------------------- |
| ⭐ **1** | **XGBoost**   | Highest accuracy, precision & overall performance |
| ⭐ **2** | Random Forest | Very strong in all metrics                        |
| ⭐ **3** | Decision Tree | Great performance but prone to overfitting        |

Insights:

* Tree-based models outperform linear models significantly
* SMOTE improved recall for minority churn class
* Churn prediction is highly dependent on spending patterns and contract length

---

# 📬 Contact

If you found this project helpful, feel free to reach out 😊

```
👤 Author: Afsaneh Shamsaddini
```

🚀 *Don’t forget to ⭐ star this repository if you like it!*

---
