# Handling Date and Time Data in Pandas

This repository contains a comprehensive guide and demonstration on how to process, extract, and manipulate date and time features in Python using Pandas and NumPy. Handling temporal data effectively is crucial for feature engineering in machine learning tasks.

---

## Overview

The primary notebook (`working-with-dates-and-time.ipynb`) demonstrates end-to-end processing of two different types of temporal datasets:
1. **Date-based Data (`orders.csv`)**: Contains transactional order records with date fields.
2. **Time-based Data (`messages.csv`)**: Contains message records with full date and time timestamps.

---

## Data Pipeline & Operations

### 1. Data Ingestion & Type Conversion
* Data is imported using `pd.read_csv()`.
* Raw string date/time columns are converted into native Pandas `datetime64` objects using `pd.to_datetime()`.

### 2. Feature Extraction from Dates
Using the `.dt` accessor, the following categorical and numeric features are extracted from date strings:
* **Year**: `dt.year`
* **Month Number**: `dt.month` (1 to 12)
* **Month Name**: `dt.month_name()` (e.g., January, August)
* **Day of Month**: `dt.day` (1 to 31)
* **Day of Week (Number)**: `dt.dayofweek` (0 = Monday, 6 = Sunday)
* **Day of Week (Name)**: `dt.day_name()` (e.g., Tuesday, Saturday)
* **Is Weekend Flag**: Uses `np.where()` with `.isin(['Sunday', 'Saturday'])` to create a binary feature (`1` for weekend, `0` for weekday).
* **Week of Year**: `dt.week` (1 to 52)
* **Quarter**: `dt.quarter` (1 to 4)
* **Semester**: Uses `np.where()` based on quarter value (`1` for Quarters 1 & 2; `2` for Quarters 3 & 4).

### 3. Feature Extraction from Timestamps
For columns containing both time and date:
* **Hour**: `dt.hour` (0 to 23)
* **Minute**: `dt.minute` (0 to 59)
* **Second**: `dt.second` (0 to 59)
* **Time Object**: `dt.time` (extracts HH:MM:SS)

### 4. Time Difference & Duration Calculations
Calculates elapsed time between a specific timestamp and the current date/time using `datetime.datetime.today()`:
* **Days Passed**: Extract total days using `.dt.days`.
* **Months Passed**: Computed via division with `np.timedelta64(1, 'M')`.
* **Seconds / Minutes / Hours Passed**: Computed via division with `np.timedelta64(1, 's')`, `'m'`, or `'h'`.

---

## Dependencies

To run the code locally, ensure you have the following Python packages installed:

```bash
pip install pandas numpy
