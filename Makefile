PY ?= python
MODEL ?= crf

APPROACH_crf := approach_1_crf
APPROACH_mbert := approach_2_mbert
APPROACH_arabert := approach_3_arabert
APPROACH_marbert := approach_4_marbert

.PHONY: help install test data eda quality train predict evaluate compare clean

help:
	@echo "Arabic NER — available targets"
	@echo "  make install            Install dependencies (pip)"
	@echo "  make test               Run the test suite"
	@echo "  make data               Load/cache the WikiANN ar dataset"
	@echo "  make eda                Execute the EDA notebook"
	@echo "  make quality            Generate the data quality report"
	@echo "  make train  MODEL=crf   Train an approach (crf|mbert|arabert|marbert)"
	@echo "  make predict MODEL=crf  Predict with a trained approach"
	@echo "  make evaluate MODEL=crf Evaluate predictions"
	@echo "  make compare            Regenerate the comparison table"
	@echo "  make clean              Remove caches"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

data:
	$(PY) data/load_data.py

eda:
	$(PY) -m jupyter nbconvert --to notebook --execute data/eda.ipynb --inplace

quality:
	$(PY) data/quality_report.py

train:
	$(PY) src/$(APPROACH_$(MODEL))/train.py --config configs/$(MODEL).yaml

predict:
	$(PY) src/$(APPROACH_$(MODEL))/predict.py --model_dir results/models/$(MODEL) --split test --out results/predictions/$(MODEL)_test.jsonl

evaluate:
	$(PY) src/evaluate.py --model $(MODEL)

compare:
	$(PY) src/compare.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
