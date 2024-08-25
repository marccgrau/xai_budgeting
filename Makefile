# Define the shell used by make
SHELL := /bin/bash

# Define the command to run Poetry
POETRY := poetry run

# python args
CATEGORY ?= "Alle"


 FILE_PATH ?= "data/final/merged_complete.csv"
# FILE_PATH ?= "data/final/merged_double_digit.csv"
#FILE_PATH ?= "data/final/merged_complete_preprocessed.csv"
#FILE_PATH ?= "data/final/merged_complete_filtered.csv"

ACC_ID ?= None
REGION ?= None
# Pipeline steps
main:
	make fetch_data
	make merge_data
	make transform_data
	make tune_and_train_catboost FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make evaluate_catboost FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make tune_and_train_xgboost FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make evaluate_xgboost FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make evaluate_ensemble FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make tune_and_train_svm FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make evaluate_svm FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make tune_and_train_lasso_svm FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make evaluate_lasso_svm FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make tune_and_train_lstm FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make tune_and_train_rforest FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make evaluate_rforest FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make tune_and_train_prophet FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make evaluate_prophet FILE_PATH=$(FILE_PATH) CATEGORY=$(CATEGORY)
	make aggregated_evaluation FILE_PATH=$(FILE_PATH)



fetch_data:
	$(POETRY) python scripts/fetch_data.py 

merge_data:
	$(POETRY) python scripts/merge_data.py

transform_data:
	$(POETRY) python scripts/transform_data.py

tune_and_train_catboost:
	$(POETRY) python src/catboost/tune_and_train.py --file_path $(FILE_PATH) --category $(CATEGORY)

evaluate_catboost:
	$(POETRY) python src/catboost/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY)

tune_and_train_xgboost:
	$(POETRY) python src/xgboost/tune_and_train.py --file_path $(FILE_PATH) --category $(CATEGORY)

evaluate_xgboost:
	$(POETRY) python src/xgboost/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY) --acc_id $(ACC_ID) --region $(REGION)

dartsEvaluationScript:
	$(POETRY) python src/dartsEvaluation/dartsEvaluationScript.py --file_path $(FILE_PATH) --category $(CATEGORY)

tune_and_train_svm:
	$(POETRY) python src/svm/tune_and_train.py --file_path $(FILE_PATH) --category $(CATEGORY)

evaluate_svm:
	$(POETRY) python src/svm/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY) --acc_id $(ACC_ID) --region $(REGION)


tune_and_train_lasso_svm:
	$(POETRY) python src/lassosvm/tune_and_train.py --file_path $(FILE_PATH) --category $(CATEGORY)

evaluate_lasso_svm:
	$(POETRY) python src/lassosvm/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY) --acc_id $(ACC_ID) --region $(REGION)
evaluate_ensemble:
	$(POETRY) python src/ensemble/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY)

tune_and_train_lstm:
	$(POETRY) python src/lstm/tune_and_train2.py --file_path $(FILE_PATH) --category $(CATEGORY)

tune_and_train_rforest:
	$(POETRY) python src/rforestregression/tune_and_train.py --file_path $(FILE_PATH) --category $(CATEGORY)

evaluate_rforest:
	$(POETRY) python src/rforestregression/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY) --acc_id $(ACC_ID) --region $(REGION)

tune_and_train_prophet:
	$(POETRY) python src/prophet/tune_and_train.py --file_path $(FILE_PATH) --category $(CATEGORY)

evaluate_prophet:
	$(POETRY) python src/prophet/evaluate.py --file_path $(FILE_PATH) --category $(CATEGORY) --acc_id $(ACC_ID) --region $(REGION)

aggregated_evaluation:
	$(POETRY) python src/xgboost/aggregeated_evaluation.py --file_path $(FILE_PATH)



# You can also add targets for installing dependencies, running tests, etc.
install_deps:
	poetry install

test:
	$(POETRY) pytest


.PHONY: fetch_data process_data install_deps test