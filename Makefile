# Define the shell used by make
SHELL := /bin/bash

# Define the command to run Poetry
POETRY := poetry run

# python args
CATEGORY ?= "Alle"
FILE_PATH ?= "data/final/merged_double_digit.csv"
ACC_ID ?= None
REGION ?= None

# Pipeline steps
main:
	make fetch_data
	make merge_data
	make transform_data
	make tune_and_train_catboost FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make evaluate_catboost FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY) ACC_ID=$(ACC_ID) REGION=$(REGION)
	make tune_and_train_xgboost FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make evaluate_xgboost FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY) ACC_ID=$(ACC_ID) REGION=$(REGION)
	make evaluate_ensemble FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)

fetch_data:
	$(POETRY) python scripts/fetch_data.py 

merge_data:
	$(POETRY) python scripts/merge_data.py

transform_data:
	$(POETRY) python scripts/transform_data.py

tune_and_train_catboost:
	$(POETRY) python src/catboost/tune_and_train.py --file_path $(FILE_PATH) --category $(CATEGORY)

evaluate_catboost:
	$(POETRY) python src/catboost/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY) --acc_id $(ACC_ID) --region $(REGION)

tune_and_train_xgboost:
	$(POETRY) python src/xgboost/tune_and_train.py --file_path $(FILE_PATH) --category $(CATEGORY)

evaluate_xgboost:
	$(POETRY) python src/xgboost/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY) --acc_id $(ACC_ID) --region $(REGION)

tune_and_train_svm:
	$(POETRY) python src/svm/tune_and_train.py --file_path $(FILE_PATH) --category $(CATEGORY)

evaluate_svm:
	$(POETRY) python src/svm/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY) --acc_id $(ACC_ID) --region $(REGION)

evaluate_ensemble:
	$(POETRY) python src/ensemble/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY)

# You can also add targets for installing dependencies, running tests, etc.
install_deps:
	poetry install

test:
	$(POETRY) pytest


.PHONY: fetch_data process_data install_deps test