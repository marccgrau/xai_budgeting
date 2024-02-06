# Define the shell used by make
SHELL := /bin/bash

# Define the command to run Poetry
POETRY := poetry run

# Pipeline steps
main:
	make fetch_data
	make merge_data
	make transform_data
	make tune_and_train_catboost
	make evaluate_catboost

fetch_data:
	$(POETRY) python scripts/fetch_data.py

merge_data:
	$(POETRY) python scripts/merge_data.py

transform_data:
	$(POETRY) python scripts/transform_data.py

tune_and_train_catboost:
	$(POETRY) python src/catboost/tune_and_train.py

evaluate_catboost:
	$(POETRY) python src/catboost/evaluate.py

tune_and_train_xgboost:
	$(POETRY) python src/xgboost/tune_and_train.py

evaluate_xgboost:
	$(POETRY) python src/xgboost/evaluate.py

evaluate_ensemble:
	$(POETRY) python src/ensemble/evaluate.py

# You can also add targets for installing dependencies, running tests, etc.
install_deps:
	poetry install

test:
	$(POETRY) pytest


.PHONY: fetch_data process_data install_deps test