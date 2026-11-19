PY ?= $(CURDIR)/.venv/bin/python
JUPYTER ?= $(CURDIR)/.venv/bin/jupyter
NB = 01_data_understanding 02_feature_engineering 03_baselines 04_ml_ranker 05_deep_learning 06_conclusion

.PHONY: install data data-light notebooks all

install:
	python3 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements.txt
	$(PY) -m ipykernel install --user --name rcaeval --display-name "RCAEval (.venv)"

data:            ## all of RE2 incl. traces (~3.5 GB)
	$(PY) scripts/download_data.py

data-light:      ## metrics + logs only (~0.5 GB)
	$(PY) scripts/download_data.py --no-traces

notebooks:
	cd notebooks && for n in $(NB); do $(JUPYTER) nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $$n.ipynb || exit 1; done

all: data notebooks
