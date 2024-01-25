# Define the shell used by make
SHELL := /bin/bash

# Define the command to run Poetry
POETRY := poetry run

# Define your targets
fetch_data:
	$(POETRY) python scripts/fetch_data.py

merge_data:
	$(POETRY) python scripts/merge_data.py

# You can also add targets for installing dependencies, running tests, etc.
install_deps:
	poetry install

test:
	$(POETRY) pytest

# Add more targets as needed

.PHONY: fetch_data process_data install_deps test