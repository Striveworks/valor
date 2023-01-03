test-env:
	docker compose --env-file .env.testing up --build -d

dev-env:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --env-file .env.testing up --build

unit-tests:
	python -m pytest -v tests/*