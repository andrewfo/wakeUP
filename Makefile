# wakeUp — reproducible pipeline targets.
# On Windows without `make`, run the underlying `python` commands directly
# (see README "Reproduce in one command").

PYTHON ?= python

.PHONY: help install data features train eval figures milestone test lint clean

help:
	@echo "Targets:"
	@echo "  install   install package + dev deps (editable)"
	@echo "  milestone run end-to-end first-milestone slice (data+attacks+baseline+figures)"
	@echo "  test      run the test suite"
	@echo "  figures   regenerate figures from the last run"
	@echo "  clean     remove generated data/figures/caches"

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Phase-scoped targets (thin wrappers over the milestone runner for now;
# split into standalone stages as later phases land).
data:
	$(PYTHON) scripts/run_milestone.py --config configs/default.yaml

features: data

train: data

eval: data

figures: milestone

milestone:
	$(PYTHON) scripts/run_milestone.py --config configs/default.yaml

# Defensible single-attack slice from the plan's "First milestone".
milestone-jump:
	$(PYTHON) scripts/run_milestone.py --config configs/default.yaml --single-attack position_jump

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests scripts

clean:
	rm -rf data/raw/* data/interim/* data/processed/* figures/*.png
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
