.PHONY: install lint test notebooks all

install:
	pip install -e ".[dev]"

lint:
	python -m ruff check src tests

test:
	python -m pytest

notebooks:
	cd notebooks && for nb in 0*.ipynb; do \
		jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 $$nb; \
	done

all: lint test notebooks
