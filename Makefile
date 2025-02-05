all: lf test-cov

test:
	pytest

test-cov:
	pytest --cov=colocationpy

format:
	ruff format

lint:
	ruff check --fix

lf: lint format
