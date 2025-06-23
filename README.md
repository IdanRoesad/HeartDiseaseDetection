# Heart Disease Prediction using Stacking Ensemble

This project demonstrates the use of a stacking ensemble model to predict the presence of heart disease based on a patient's medical attributes. The primary goal is to showcase how combining multiple machine learning models can lead to improved classification accuracy.

The script is built to run in a Google Colab environment and compares the performance of the stacking classifier using different algorithms as the final meta-estimator.

## Overview

The project follows a machine learning workflow designed for demonstration and comparison:
1.  **Data Loading:** Loads the heart disease dataset from a zip file.
2.  **Model Definition:** Defines a set of diverse base models (or "base learners") to be used in the ensemble.
3.  **Data Splitting:** Splits the dataset into training and testing sets.
4.  **Comparative Modeling:** A **Stacking Classifier** is constructed and trained multiple times. In each run, a different algorithm is used as the final "meta-classifier" to see how it affects the overall accuracy.
5.  **Evaluation:** The accuracy score for each stacking configuration is printed to the console, allowing for a direct comparison of the results.

## Methodology: Stacking Classifier

A Stacking (Stacked Generalization) Classifier is an ensemble learning technique that leverages the strengths of multiple models. The predictions from the base models are used as input for a final meta-model, which makes the ultimate prediction.

The architecture used in this project is as follows:

* **Level 0 Models (Base Learners):**
    1.  `LogisticRegression`
    2.  `DecisionTreeClassifier`
    3.  `RandomForestClassifier`
    4.  `SVC` (Support Vector Classifier)
    5.  `KNeighborsClassifier`

* **Level 1 Models (Meta-Estimators Compared):**
    * The script tests the performance of the stacking ensemble by using each of the following as the final estimator:
        1.  `DecisionTreeClassifier`
        2.  `LogisticRegression`
        3.  `KNeighborsClassifier`

This approach allows for an effective comparison to see which meta-classifier works best for combining the predictions of the base learners on this specific dataset.

## Dependencies

This project relies on the following Python libraries. You can install them using pip:

pip install numpy pandas matplotlib seaborn scikit-learn


## How to Run

This script is designed to be run in a **Google Colab** notebook.

1.  **Upload to Colab:** Open Google Colab and upload the `demo_ml_stacking.py` file.
2.  **Upload Data:** Upload the `data_heart.zip` file to your Colab environment.
3.  **Run All Cells:** Execute the cells in the notebook from top to bottom.
    * The script will unzip the data, split it, and then train and evaluate the stacking classifier with each of the specified meta-estimators.
4.  **Observe Results:** The final output will be the accuracy scores printed for each configuration (Stacking with Decision Tree, Stacking with Logistic Regression, etc.), allowing you to compare their performance directly.

## File Description

* **`demo_ml_stacking.py`**: The main Python script containing all code for data loading, model training, and evaluation.
* **`data_heart.zip`**: The compressed dataset containing patient attributes and the target variable indicating the presence or absence of heart disease.
