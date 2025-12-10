MATH5472 Final Project — What If Without Random Forest

This repository contains the implementation and experiments for the MATH5472 course project.  
The objective is to evaluate the contribution of Random Forest by comparing it against baseline models on the Wine Quality dataset.

1. Project Overview
-------------------

Course: MATH5472 – Statistical Machine Learning  
Topic: What If Without the Random Forest Method  
Dataset: Wine Quality (red and white variants)  
Task: Binary classification — predict whether a wine is “good” (quality ≥ 7)

NABC Summary:

Need:
Linear models underfit nonlinear relationships; single decision trees have high variance.

Approach:
Random Forest using bootstrap sampling and random feature subsets, compared with Logistic Regression, Decision Tree, and Gradient Boosting.

Benefit:
Random Forest provides the strongest accuracy, F1-score, and ROC-AUC.

Competitors:
Logistic Regression, Decision Tree, Gradient Boosting.

2. Repository Structure
-----------------------

MATH5472-final-project/
│
├── data/
│     ├── winequality-red.csv
│     └── winequality-white.csv
│
├── src/
│     ├── __init__.py
│     ├── config.py
│     ├── data_utils.py
│     ├── train_and_evaluate.py
│     └── plot_results.py
│
├── results/
│     ├── metrics.csv
│     ├── confusion_matrix_logistic_regression.png
│     ├── confusion_matrix_decision_tree.png
│     ├── confusion_matrix_random_forest.png
│     ├── confusion_matrix_gradient_boosting.png
│     ├── feature_importance_random_forest.png
│     ├── feature_importance_gradient_boosting.png
│     └── model_comparison_*.png
│
├── .gitignore
└── paper.tex

3. Environment Setup
--------------------

cd MATH5472-final-project  
python3 -m venv .venv  
source .venv/bin/activate  

pip install pandas scikit-learn matplotlib  

4. Data Placement
-----------------

Place these files into the data/ directory:

winequality-red.csv  
winequality-white.csv  

5. Running the Experiments
--------------------------

Train and evaluate all models:

python -m src.train_and_evaluate

The script loads data, trains Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, evaluates all models, and saves results.

Generate comparison plots:

python -m src.plot_results

6. Preliminary Results
----------------------

Logistic Regression: accuracy ≈ 0.83, ROC-AUC ≈ 0.81  
Decision Tree: accuracy ≈ 0.83, ROC-AUC ≈ 0.75  
Random Forest: accuracy ≈ 0.89, ROC-AUC ≈ 0.92  
Gradient Boosting: accuracy ≈ 0.85, ROC-AUC ≈ 0.87  

7. Reproducibility
------------------

git clone https://github.com/MichaelYangzk/MATH5472-final-project.git  
cd MATH5472-final-project  
python3 -m venv .venv  
source .venv/bin/activate  
pip install pandas scikit-learn matplotlib  
python -m src.train_and_evaluate  
python -m src.plot_results
