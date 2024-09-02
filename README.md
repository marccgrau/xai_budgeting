
# Artificial Intelligence to Improve Public Budgeting 
### Software to ICIS submission

Santschi, Dominic; Grau, Marc; Fehrenbacher, Dennis; Blohm, Ivo

## Project Description
XAI Budgeting focuses on forecasting realized values for accounts in public administration, with a specific focus on Swiss cantons. This project analyzes a range of financial metrics including expenditures, expenses, and assets, using public data that adheres to HRM1 and HRM2 standards. The analysis covers data from 2010 onwards, including financial documents like income statements and balance sheets. The goal is to deliver accurate and explainable predictions to support fiscal planning and analysis.

## Overview

This project features a data processing and machine learning pipeline automated with a Makefile. The pipeline encompasses steps for data fetching, merging, transforming, model tuning and training using both CatBoost and XGBoost models, and evaluating these models' performance. It also includes the capability to construct a simple average ensemble model for improved prediction accuracy.

## Dependencies
To install all necessary dependencies, ensure you have Poetry installed on your system.

Command to install dependencies:
```shell
make install_deps
```

## Pipeline Steps

### Fetching Data
Fetches necessary data for the project.
```shell
make fetch_data
```

### Merging Data
Merges fetched data into a unified format suitable for analysis.
```shell
make merge_data
```

### Transforming Data
Applies required transformations to prepare the data for modeling.
```shell
make transform_data
```

### Tuning and Training Models
Tunes hyperparameters and trains models using both CatBoost and XGBoost algorithms. These steps allow specifying a `FILE_PATH` to the dataset and a `CATEGORY` for targeted feature engineering.

#### CatBoost Model
```shell
make tune_and_train_catboost FILE_PATH="path/to/your/data.csv" CATEGORY="YourCategory"
```

#### XGBoost Model
```shell
make tune_and_train_xgboost FILE_PATH="path/to/your/data.csv" CATEGORY="YourCategory"
```

### Evaluating Models
Evaluates the performance of the trained models against actual data and budget figures. Supports specifying `FILE_PATH` and `CATEGORY` for targeted evaluation.

#### CatBoost Model
```shell
make evaluate_catboost FILE_PATH="path/to/your/data.csv" CATEGORY="YourCategory"
```

#### XGBoost Model
```shell
make evaluate_xgboost FILE_PATH="path/to/your/data.csv" CATEGORY="YourCategory"
```

### Evaluating Ensemble Model
Evaluates the performance of a simple average ensemble constructed from the CatBoost and XGBoost model predictions.
```shell
make evaluate_ensemble FILE_PATH="path/to/your/data.csv" CATEGORY="YourCategory"
```

### Additional Commands

#### Running Tests
Runs tests to ensure the pipeline's components are functioning as expected.
```shell
make test
```

This README outlines the steps to utilize the automated pipeline for financial forecasting within the XAI Budgeting project. It specifies how to run each component of the pipeline, including fetching and processing data, tuning and training models, and evaluating their performance.
