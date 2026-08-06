# Data Preprocessing: Power Transformations with Scikit-Learn

A comprehensive guide and practical implementation demonstrating how to fix skewed feature distributions using **Power Transformations** (`PowerTransformer` from `scikit-learn`). This repository explores the mathematical mechanisms, practical constraints, and visual diagnostics for normalizing continuous features prior to machine learning model training.

---

## 1. Project Overview

Many parametric machine learning algorithms (such as Linear Regression, Logistic Regression, and Linear Discriminant Analysis) assume that numerical input features follow a Gaussian (Normal) bell-curve distribution. Skewed features can bias model estimates and degrade performance.

This project implements and compares two primary power transformation techniques:
1. **Box-Cox Transformation**: An optimal power transform for strictly positive data.
2. **Yeo-Johnson Transformation**: An extension of Box-Cox designed to support zero and negative values.

---

## 2. Theoretical Background & Mathematical Comparison

Unlike basic functional transformations (such as manual logarithms or square root scaling), Scikit-Learn's `PowerTransformer` estimates an optimal parameter ($\lambda$) using Maximum Likelihood Estimation (MLE) to dynamically force data toward a normal distribution.

### Box-Cox Transformation
Defined mathematically as:

$$y^{(\lambda)} = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln(x) & \text{if } \lambda = 0 \end{cases}$$

* **Primary Use Case**: Features with heavy right-skew where all observations are strictly greater than zero ($x > 0$).
* **Constraint**: Raises a runtime exception if applied to zero or negative values.

### Yeo-Johnson Transformation
Extends Box-Cox to handle non-positive real numbers ($x \le 0$):

$$y^{(\lambda)} = \begin{cases} \frac{(x + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, x \ge 0 \\ \ln(x + 1) & \text{if } \lambda = 0, x \ge 0 \\ -\frac{(-x + 1)^{2 - \lambda} - 1}{2 - \lambda} & \text{if } \lambda \neq 2, x < 0 \\ -\ln(-x + 1) & \text{if } \lambda = 2, x < 0 \end{cases}$$

* **Primary Use Case**: Features containing zero or negative values (e.g., net profit/loss, temperature, financial gains/losses).
* **Constraint**: None; operates across all real numbers.

---

## 3. Comparative Summary Matrix

| Metric / Feature | Box-Cox | Yeo-Johnson |
| :--- | :--- | :--- |
| **Scikit-Learn API** | `PowerTransformer(method='box-cox')` | `PowerTransformer(method='yeo-johnson')` |
| **Supports $x = 0$** | No | Yes |
| **Supports $x < 0$** | No | Yes |
| **Domain Range** | $x \in (0, \infty)$ | $x \in (-\infty, \infty)$ |
| **Parameter Optimization** | Automated via MLE | Automated via MLE |
| **Default Standard Scaling** | Mean = 0, Std = 1 (`standardize=True`) | Mean = 0, Std = 1 (`standardize=True`) |

---

## 4. Repository Structure

```text
.
├── Power-Transformer.ipynb    # Jupyter Notebook containing code and visualizations
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies
