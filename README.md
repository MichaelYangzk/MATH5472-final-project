# MATH5472 Final Project: What If Without Random Forest

This repository contains the implementation and experimental analysis for the MATH5472 course project. The goal is to evaluate the contribution of Random Forest by comparing it against baseline models using the Wine Quality dataset.

## 1. Project Overview

- Course: MATH5472 – Statistical Machine Learning  
- Topic: What If Without the Random Forest Method  
- Dataset: Wine Quality (red and white variants)  
- Task: Binary classification of wine quality  
- Objective: Assess performance degradation when Random Forest is removed.

The project follows the NABC structure:

- **Need**: Linear models underfit nonlinear relationships. Single decision trees have high variance and instability.  
- **Approach**: Random Forest with bootstrap sampling and random feature selection. Comparison against Logistic Regression, Decision Tree, and Gradient Boosting.  
- **Benefit**: Random Forest achieves higher ROC-AUC, F1-score, and accuracy.  
- **Competitors**: Logistic Regression, Decision Tree, Gradient Boosting.

## 2. Repository Structure

\`\`\`
MATH5472-final-project/
│
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── train_and_evaluate.py
│   └── plot_results.py
│
├── results/
│   ├── metrics.csv
│   ├── confusion_matrix_*.png
│   ├── feature_importance_*.png
│   └── model_comparison_*.png
│
├── .gitignore
└── paper.tex
\`\`\`

## 3. Environment Setup

### Create virtual environment

\`\`\`
cd MATH5472-final-project
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

### Install dependencies

\`\`\`
pip install pandas scikit-learn matplotlib
\`\`\`

## 4. Data

Place the following files under the `data/` directory:

- winequality-red.csv  
- winequality-white.csv  

## 5. Running the Experiments

Run:

\`\`\`
python -m src.train_and_evaluate
\`\`\`

The script:

1. Loads and preprocesses data  
2. Trains Logistic Regression, Decision Tree, Random Forest, Gradient Boosting  
3. Evaluates models  
4. Writes metrics and plots to the `results/` directory

### Generate comparison plots

\`\`\`
python -m src.plot_results
\`\`\`

## 6. Preliminary Results

Approximate performance:

- Logistic Regression: accuracy ≈ 0.83, ROC-AUC ≈ 0.81  
- Decision Tree: accuracy ≈ 0.83, ROC-AUC ≈ 0.75  
- Random Forest: accuracy ≈ 0.89, ROC-AUC ≈ 0.92  
- Gradient Boosting: accuracy ≈ 0.85, ROC-AUC ≈ 0.87  

## 7. Reproducibility

\`\`\`
git clone https://github.com/MichaelYangzk/MATH5472-final-project.git
cd MATH5472-final-project
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn matplotlib
python -m src.train_and_evaluate
python -m src.plot_results
\`\`\`

