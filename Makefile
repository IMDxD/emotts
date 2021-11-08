requirements:
	pip install -r requirements-dev.txt

start: requirements

pytest:
	pytest --cov=src --cov-fail-under 80 --blockage  --cov-report term-missing

coverage-collect:
	coverage run -m pytest

coverage-report:
	coverage html

coverage: coverage-collect coverage-report

mypy:
	mypy .

flake8:
	flake8 src .

isort:
	isort .

check: isort flake8 mypy pytest
