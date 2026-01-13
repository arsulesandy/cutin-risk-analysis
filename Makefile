.PHONY: install test lint format

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	echo "Add linting commands (e.g., ruff, black)."

format:
	echo "Add formatting commands (e.g., black, isort)."
