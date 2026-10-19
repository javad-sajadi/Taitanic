# Titanic Disaster: Survival Prediction & Feature Engineering Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-darkblue.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and exploratory data analysis pipeline predicting passenger survival for the classic **Kaggle Titanic: Machine Learning from Disaster** competition.

---

## 📌 Project Overview

This repository demonstrates rigorous exploratory data analysis (EDA), predictive feature engineering, and cross-validated machine learning modeling on tabular historical passenger data.

### Engineered Features & Transformations
- **Title Extraction:** Parsed honorifics (`Mr`, `Miss`, `Mrs`, `Master`, `Rare`) from passenger names to infer social status and demographics.
- **Family Structure:** Created compound interaction features (`FamilySize = SibSp + Parch + 1`, `IsAlone`, `LargeFamily`).
- **Conditional Median Imputation:** Imputed missing values in `Age` conditional on passenger `Title`, and missing values in `Fare` conditional on `Pclass`.
- **Categorical Encodings:** Multi-class dummy variables for embarkation port and passenger class.

---

## 📊 Model Benchmarking

Evaluated across multiple classical classifiers using 5-Fold Stratified Cross-Validation:

| Model | Cross-Validation Accuracy | Characteristics |
| :--- | :---: | :--- |
| **Random Forest Classifier** | **~83.2%** | Non-linear tree ensemble capturing feature interactions |
| **Support Vector Machine (RBF)** | **~82.4%** | Kernelized margin maximization on standardized features |
| **Logistic Regression** | **~80.5%** | L2-regularized linear baseline with odds-ratio interpretability |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/javad-sajadi/Taitanic.git
cd Taitanic
pip install -r requirements.txt
```

### 2. Run the Automated Pipeline
Preprocesses data, benchmarks all models, and generates `submission.csv`:
```bash
python titanic_pipeline.py
```

### 3. Interactive Jupyter Notebook
Explore the interactive visual analysis, correlation matrices, and model experiments:
```bash
jupyter notebook Titanic.ipynb
```

---

## 📁 Repository Structure

```
.
├── titanic/
│   ├── train.csv             # Kaggle training dataset (891 passenger records)
│   ├── test.csv              # Kaggle test evaluation dataset (418 records)
│   └── gender_submission.csv # Baseline gender benchmark submission
├── Titanic.ipynb             # Interactive exploratory data analysis & experiments
├── titanic_pipeline.py       # Production-ready CLI pipeline with CV benchmarking
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore configuration
```

---

## 📜 Author & License
- **Author:** [Javad Sajadi](https://github.com/javad-sajadi)
- **License:** MIT License
