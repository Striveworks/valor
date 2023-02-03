test-env:
	docker compose -p velour --env-file .env.testing up --build -d

dev-env:
	docker compose -p velour -f docker-compose.yml -f docker-compose.dev.yml --env-file .env.testing up --build

stop-env:
	docker compose -p velour down

unit-tests:
	python -m pytest -v tests/unit-tests

start-postgis-docker:
	docker run -p 5432:5432 -e POSTGRES_PASSWORD=password -d postgis/postgis

functional-tests:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost pytest -v tests/functional-tests