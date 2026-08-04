# 🛒 E-Commerce Customer Return Prediction

A end-to-end Machine Learning pipeline designed to predict customer product returns (`returned = 1`) using transaction and demographic features. 

This project demonstrates practical techniques in synthetic data generation, handling missing data (MCAR/MAR), categorical encoding, and preparing tabular datasets for binary classification models.

---

## 📌 Project Overview & Problem Statement

Product returns are a major cost driver in e-commerce logistics. The objective of this project is to build a predictive binary classification model that identifies high-risk return transactions, enabling proactive business decisions (e.g., targeted customer support, size/spec checks, or custom return policies).

### **Key Highlights & Technical Learning Outcomes:**
* **Synthetic Data Engineering:** Programmatically constructed a non-trivial 500-sample dataset using `numpy` probability distributions (`p=[...]`) to simulate real-world e-commerce behavior (e.g., missingness patterns, device preference skews).
* **Missing Value Analysis:** Implemented realistic missing values (`np.nan`) in features like `customer_age` (10% missing) and `discount_percentage` (5% missing) to practice robust data imputation strategies.
* **Feature Representation:** Processed a mix of numerical, ordinal, and nominal categorical features (`device_type`, `product_category`, `is_first_time_buyer`).

---

## 📊 Dataset Schema

The generated dataset (`ecommerce_returns.csv`) consists of **500 rows and 7 features**:

| Feature Name | Type | Description / Simulated Range |
| :--- | :--- | :--- |
| `customer_age` | Continuous / Categorical | Customer age (18, 22, 35, 45, 60) with simulated missing values (`NaN`). |
| `device_type` | Categorical (Nominal) | Device used for purchase (`Mobile`, `Desktop`, `Tablet`, `NaN`). |
| `product_category` | Categorical (Nominal) | Item category (`Electronics`, `Clothing`, `Home`, `Beauty`). |
| `item_price` | Continuous (Float) | Uniformly distributed price values between **$10.00** and **$500.00**. |
| `discount_percentage` | Discrete / Continuous | Applied discount percentage (0%, 10%, 20%, 50%, or `NaN`). |
| `is_first_time_buyer` | Categorical (Binary) | Whether the customer is a first-time buyer (`Yes` / `No`). |
| **`returned`** *(Target)* | Binary (0 / 1) | **Target Variable**: `1` = Item Returned, `0` = Item Kept. Imbalanced distribution (80% kept, 20% returned). |

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`
* **Environment:** JupyterLab / Jupyter Notebooks

---

## 🚀 How to Run & Reproduce

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/e-commerce-return-prediction.git](https://github.com/your-username/e-commerce-return-prediction.git)
   cd e-commerce-return-prediction
