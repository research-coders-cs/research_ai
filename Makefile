SHELL := /bin/bash

ci:
	pipenv install
	make test

DL_ASSETS := https://github.com/research-coders-cs/research_thyroid/releases/download/assets-0.1
DATASET_ZIP := Dataset_train_test_val.rar
Dataset_train_test_val:
	curl -O -L $(DL_ASSETS)/$(DATASET_ZIP)
	unrar x $(DATASET_ZIP)
net_debug.pth:
	curl -O -L $(DL_ASSETS)/$@

test: Dataset_train_test_val net_debug.pth
	rm -rf result && mkdir result
	@echo "(first time, make sure \`pipenv install\`)"
	pipenv run python3 test.py

