#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = deep-learning-cancer-classifier
PYTHON_VERSION = 3.11.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run training in cluster mode SLURM
.PHONY: train_cluster
train_cluster:
	@echo ">>> Running training in cluster mode"
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; workon $(PROJECT_NAME); else workon.bat $(PROJECT_NAME); fi"
	@sbatch --job-name=$(PROJECT_NAME) --output=logs/%j.out --error=logs/%j.err --time=24:00:00 --ntasks=1 --cpus-per-task=4 --mem=16G --wrap='$(PYTHON_INTERPRETER) cancer_classifier/train.py'
	@echo ">>> Training job submitted to SLURM. Check logs/ for output."


## Run training in local mode
.PHONY: train_local
train_local:
	@echo ">>> Running training in local mode"
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; workon $(PROJECT_NAME); else workon.bat $(PROJECT_NAME); fi"
	@$(PYTHON_INTERPRETER) cancer_classifier/train.py
	@echo ">>> Training job completed. Check logs/ for output."


## Run training in local mode with GPU
.PHONY: train_local_gpu
train_local_gpu:
	@echo ">>> Running training in local mode with GPU"
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; workon $(PROJECT_NAME); else workon.bat $(PROJECT_NAME); fi"
	@$(PYTHON_INTERPRETER) cancer_classifier/train.py --use-gpu
	@echo ">>> Training job completed. Check logs/ for output."





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) cancer_classifier/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
