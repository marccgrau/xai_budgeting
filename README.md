# XAI Budgeting

## Project Description
XAI Budgeting is focused on forecasting realized values of accounts in public administration, specifically targeting Swiss cantons. The project encompasses a range of financial metrics including expenditures, expenses, and assets, extracted from public data adhering to HRM1 and HRM2 standards. The temporal scope of the data spans from the year 2010 onwards, incorporating financial documents such as income statements and balance sheets. The aim is to provide accurate and explainable predictions that can aid in fiscal planning and analysis.

## Overview

This project implements a data processing and machine learning pipeline using a Makefile for automation. The pipeline includes steps for fetching data, merging, transforming, tuning and training a CatBoost model, and finally evaluating the model's performance.

## Dependencies
Install all necessar dependencies, make sure you have poetry installed. 

Command:
```shell
make install_deps
```

## Pipeline Steps

### 1. `fetch_data`
Fetches the necessary data for the project.

Command:
```shell
make fetch_data
```

### 2. `merge_data`
Merges the fetched data into a unified format.

Command:
```shell
make merge_data
```

### 3. `transform_data`
Applies necessary transformations to the data to prepare it for modeling.

Command:
```shell
make transform_data
```

### 4. `tune_and_train_catboost`
Tunes the hyperparameters of the CatBoost model and trains the final model based on the best setting.

Command:
```shell
make tune_and_train_catboost
```

### 5. `evaluate_catboost`
Evaluates the performance of the best CatBoost model compared to budgets. 

Command:
```shell
make evaluate_catboost
```
