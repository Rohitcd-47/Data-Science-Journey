# Telco Customer Churn Intelligence & Prediction Pipeline

## Executive Summary

Customer attrition directly degrades Net Revenue Retention (NRR) and inflates Customer Acquisition Costs (CAC). Retaining an existing customer is significantly cheaper than acquiring a new one. 

This project provides an end-to-end Machine Learning pipeline and an interactive deployment system designed to predict customer churn risk in real-time. By analyzing demographic, transactional, and service-usage patterns, the system enables retention teams to proactively intervene before a customer cancels their subscription.

---

## Model Performance & Accuracy

Because misclassifying a churning customer as "safe" leads to lost revenue, model evaluation prioritized minimizing False Negatives alongside overall accuracy:

* **Overall Model Accuracy:** ~80%
* **Primary Optimization Metric:** Recall & ROC-AUC (Ensuring high sensitivity toward at-risk customers)
* **Baseline Comparison:** Outperformed standard dummy baselines by capturing high-risk churn signals effectively across complex feature interactions.

---

## Core Features & Business Value

* **Automated Data Pipeline:** Handles raw customer data ingestion, type casting, missing value handling, and categorical encoding.
* **Predictive Intelligence:** Leverages classification algorithms to generate precise churn probability scores.
* **Interactive Dashboard:** Built with Streamlit for non-technical stakeholders, allowing real-time single-customer risk evaluations.
* **Actionable Insights:** Translates raw numerical predictions into clear retention risk tiers.

---

## Technical Stack

* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Model Training & Pipeline:** Scikit-Learn
* **Application Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud

---

## Key Terms & Domain Guide (In Hinglish)

To help recruiters and non-technical stakeholders understand the target output and input fields used in the prediction model, here is a simple breakdown of key telecom domain terms:

### What is Churn?
* **Meaning:** Jab koi customer aapki company ki service ya subscription use karna band kar deta hai aur kisi competitor ke paas chala jata hai ya plan cancel kar deta hai, toh use **Churn** kehte hain.
* **Example:** Agar aap Jio ya Airtel ka SIM card use kar rahe the aur aapne service se dissatisfied hokar Port-out karva liya ya recharge karna band kar diya, toh company ke liye aap **Churned Customer** ho gaye.

---

### Input Features Breakdown

### 1. Dependents
* **Meaning:** Kya customer ke upar koi family member financially dependent hai (jaise bachhe ya elderly parents)?
* **Example:** Agar customer married hai aur uske do bachhe hain jo uske bill par depend karte hain, toh value **Yes** hogi. Usually, dependents wale customers churn kam karte hain kyunki family setups change karna difficult hota hai.

### 2. Senior Citizen
* **Meaning:** Kya customer ki umar 60-65 saal se zyada hai?
* **Example:** Agar koi retired person service use kar raha hai, toh status **1** (Yes) hoga, warna **0** (No). Senior citizens ki service needs aur price sensitivity differ karti hai.

### 3. Tenure
* **Meaning:** Customer kitne mahino se company ki services use kar raha hai.
* **Example:** Agar kisi ne 2 saal pehle connection liya tha, toh uska tenure **24 months** hoga. High tenure normally indicates higher brand loyalty.

### 4. Partner
* **Meaning:** Kya customer married hai ya kisi relationship mein hai?
* **Example:** Single individual ke liye **No**, jabki spouse ke saath rehne wale ke liye **Yes**.

### 5. Online Security
* **Meaning:** Kya customer ne internet connection ke saath extra cybersecurity protection service li hai?
* **Example:** Antivirus ya firewall protection addon. Agar internet service hi nahi hai, toh yeh **No internet service** marked hota hai.

### 6. Online Backup
* **Meaning:** Company ka cloud storage add-on service jahan customer apna data backup kar sakta hai.
* **Example:** Like Google Drive ya iCloud space provided by the ISP. Agar user files cloud par store karta hai, toh **Yes**.

### 7. Device Protection
* **Meaning:** Hardware risk cover ya insurance for devices provided by the company.
* **Example:** Router ya setup box kharab hone par free replacement ya repair plan.

### 8. Tech Support
* **Meaning:** Priority customer support service to fix technical bugs or connection issues quickly.
* **Example:** Dedicated helpline access for immediate troubleshooting without long queue wait times.

### 9. Streaming TV & Streaming Movies
* **Meaning:** ISP ke side se premium TV channels ya OTT movie streaming subscriptions ki service bundle.
* **Example:** Network plan ke saath Netflix, Hotstar, ya Live TV ka access included hona.

### 10. Paperless Billing
* **Meaning:** Physical paper bill ki jagah email ya app par digital invoice receive karna.
* **Example:** Monthly bill PDF email par aana vs ghar par post se aana.

---

## Machine Learning Workflow

1. **Exploratory Data Analysis (EDA):** Identified key drivers behind churn, such as contract types (Month-to-month vs Long-term) and Total/Monthly Charges.
2. **Preprocessing:** Handled object-to-numeric type conversion for fields like `TotalCharges`, applied encoding to categorical variables, and scaled numerical features.
3. **Model Selection:** Evaluated multiple standard classifiers focusing on recall and ROC-AUC metrics to minimize false negatives (missing a customer who is about to churn).
4. **Deployment:** Bundled preprocessing and inference steps into a clean Streamlit Web App interface.

---

## Setup and Local Run

### Prerequisites
Ensure Python 3.8+ is installed on your machine.

### Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/CustomerChurn-Pipeline-System.git](https://github.com/your-username/CustomerChurn-Pipeline-System.git)

# Navigate to the project directory
cd CustomerChurn-Pipeline-System

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit Application
streamlit run app.py
