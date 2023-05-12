# Path: Makefile

test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint .\training --recursive=y
	pylint .\tests --recursive=y
