# 🧪 QSAR Toxicity Classification 

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains a machine learning pipeline designed to predict the toxicity of molecules based on their Quantitative Structure-Activity Relationship (QSAR) descriptors. The primary objective of this project is to accurately classify a compound as **Toxic** or **NonToxic** using a heavily optimized Random Forest ensemble model.

## 📊 The Dataset & Challenges
The dataset (`data.csv`) consists of chemical and physical properties of molecules. 
* **Samples:** 171 molecules
* **Features:** 1,203 molecular descriptors
* **Target:** `Class` (Binary: Toxic vs. NonToxic)

### Key Data Science Challenges Addressed:
1. **The Curse of Dimensionality (P >> N):** With over 1,200 features and only 171 samples, the data is highly susceptible to overfitting. This was mitigated using rigorous mathematical feature selection (ANOVA F-test).
2. **Class Imbalance:** The dataset has roughly a 2:1 ratio of NonToxic (115) to Toxic (56) molecules. This was addressed natively in the model using balanced class weights to heavily penalize minority-class misclassifications.

## ⚙️ Modeling Strategy & Pipeline
To prevent **Data Leakage** during cross-validation, the entire workflow is encapsulated within a strict Scikit-Learn `Pipeline`. 

The steps execute in the following order:
1. **Preprocessing:** `StandardScaler()` normalizes all 1,200+ features to a mean of 0 and a standard deviation of 1.
2. **Feature Selection:** `SelectKBest(score_func=f_classif)` mathematically isolates the most statistically significant chemical descriptors, aggressively reducing the dimensionality.
3. **Classification:** `RandomForestClassifier(class_weight='balanced')` builds a robust ensemble of decision trees to capture non-linear chemical interactions.

**Hyperparameter Tuning:** `GridSearchCV` was deployed across a 5-Fold Stratified Cross-Validation setup to test 72 distinct mathematical configurations. The grid optimized the number of selected features (k), the number of estimators, and the maximum depth of the trees to maximize the **Balanced Accuracy** metric.

## 📈 Key Results
The model was evaluated on a strictly held-out 20% test set. 

* **Best Cross-Validation Balanced Accuracy:** ~47.25%
* **Final Test Accuracy:** 60%
* **NonToxic Recall:** 83% 

*Note: While the overall accuracy is 60%, the stringent pipeline successfully prevented the model from artificially inflating its score through data leakage, presenting a realistic baseline for $P \gg N$ high-dimensional chemical data.*

## 🚀 How to Run the Project

### Running via Google Colab (Recommended)
1. Clone this repository or download the `.ipynb` file.
2. Upload the notebook to [Google Colab](https://colab.research.google.com/).
3. Upload the `data.csv` file directly into your Colab session storage.
4. Select `Runtime` > `Run all`.

### Running Locally (Debian/Linux)
Ensure you have Python installed, then set up your environment:
```bash
# Clone the repository
git clone <your-repository-url>
cd <your-repository-folder>

# Install the required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the Jupyter Notebook
jupyter notebook
