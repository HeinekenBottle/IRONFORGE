.PHONY: setup fmt lint type test precommit

setup:
	pip install -U pip
	pip install -e .[dev]
	pre-commit install

fmt:
	black ironforge tests

lint:
	ruff check ironforge tests

type:
	mypy ironforge

test:
	pytest -q

precommit:
	pre-commit run --all-files