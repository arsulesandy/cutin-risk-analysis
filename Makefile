PYTHON ?= python3

.PHONY: install test lint format check phases clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

check: test lint

phases:
	$(PYTHON) scripts/run_thesis_phases.py --list-phases

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
