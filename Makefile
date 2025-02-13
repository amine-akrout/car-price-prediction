# Path: Makefile

test:
	pytest tests/

quality_checks:
	ruff check . --fix

train:
	cd training && python model_training.py

deploy-prefect:
	docker-compose up --build


serve-api:
	cd serving && docker-compose up --build

stop:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	rm -rf training/model/*
	rm -rf training/mlruns/*
