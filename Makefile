# Path: Makefile

test:
	pytest training/tests/

quality_checks:
	isort .
	black .
	pylint .\training --recursive=y --fail-under=9

train:
	cd training && python model_training.py

deploy:
	docker-compose up -d

serve:
	cd serving && docker-compose build && docker-compose up -d

stop:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	rm -rf training/model/*
	rm -rf training/mlruns/*
