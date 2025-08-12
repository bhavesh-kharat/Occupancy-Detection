# Occupancy Detection Machine Learning Project

**Room occupancy** is a classic smart‑building problem: when you know how many people are in a space you can save energy, optimise ventilation and plan resources.  This project uses a mixture of temperature, light, sound, CO₂ and motion sensors to build models that estimate occupancy from raw sensor data.

## Overview

The notebook in this repository walks through a complete machine‑learning workflow: loading a public dataset, exploring and cleaning the data, engineering features and training several predictive models.  The dataset contains **18 sensor features** (four temperature, four light, four sound sensors, a CO₂ level and slope, and two motion sensors) plus a target value `Room_Occupancy_Count` indicating how many people were present.  There are 10 129 rows and 19 columns in total, timestamped by date and time.

## Objectives

We keep the goals simple:

* **Predict occupancy counts** based on the sensor readings.
* **Compare modelling approaches** ranging from simple linear models to ensembles and a small convolutional neural network.
* **Understand feature importance** by examining which sensors contribute most to accurate predictions.

## Workflow summary

1. **Data preparation:** import the CSV file, inspect its shape and column names, and handle any missing values.  Continuous variables are standardised or scaled, and sampling techniques (over‑/under‑sampling) are used to balance the classes.
2. **Exploratory analysis:** visualise distributions and correlations to identify redundant features; temperature and light sensors often track each other, while motion sensors provide distinct signals.
3. **Modelling:** split the data into training and test sets and train a variety of models: logistic regression, k‑nearest neighbours, support vector machines, decision trees, random forests, boosting (Gradient/AdaBoost), linear discriminant analysis, Gaussian Naïve Bayes and a compact convolutional neural network.
3. **Evaluation:** use cross‑validation and metrics such as accuracy, precision, recall and F1‑score to compare models.  Ensemble methods (Random Forest and Gradient Boosting) tend to perform best, with the CNN offering competitive results but requiring more tuning.

## Running the notebook

To reproduce the analysis, open `OccupancyDetection.ipynb` in a Jupyter environment.  Make sure you have the required Python packages (Pandas, scikit‑learn, imbalanced‑learn, TensorFlow, NumPy and Matplotlib) installed.  Run the cells from top to bottom; the dataset will download automatically from the UCI repository.
