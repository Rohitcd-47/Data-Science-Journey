# 🧪 Machine Learning: Preprocessing with Function Transformers

A hands-on, step-by-step practice guide exploring **Function Transformers** (`sklearn.preprocessing.FunctionTransformer`) in Python. This project demonstrates how mathematical transformations can fix skewed data distributions to bring them closer to a Gaussian (Normal) distribution.

---

## 📌 Features & Topics Covered

- 📐 **Log Transformation (ln(1+x))**: Squishing extreme right-skewed feature distributions (e.g., California Housing `Population`).
- 🟩 **Square Transformation ($x^2$)**: Adjusting left-skewed features (e.g., Customer Satisfaction Scores).
- 🔄 **Reciprocal Transformation ($\frac{1}{x}$)**: Handling extreme right-skewed ratios and fractions (e.g., Bedrooms-per-Room ratio).
- 📊 **Visual Diagnostics**: Comparative analysis using Seaborn **KDE plots** and SciPy **Q-Q (Quantile-Quantile) plots**.

---

## 🛠️ Tech Stack & Requirements

- **Python 3.8+**
- **JupyterLab / Jupyter Notebook**

### Libraries Used
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
