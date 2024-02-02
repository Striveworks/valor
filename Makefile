test-env:
	docker compose -p velour --env-file ./api/.env.testing up --build -d

dev-env:
	docker compose -p velour -f docker-compose.yml --env-file ./api/.env.testing up --build -d

stop-env:
	docker compose -p velour down

unit-tests:
	python -m pytest -v ./api/tests/unit-tests
	python -m pytest -v ./client/unit-tests

start-postgis-docker:
	docker run -p 5432:5432 -e POSTGRES_PASSWORD=password -e POSTGRES_DB=velour -d docker.io/postgis/postgis

run-migrations:
	docker build -f=migrations/Dockerfile ./migrations -t migrations && \
	docker run -e POSTGRES_PASSWORD=password -e POSTGRES_HOST=localhost -e POSTGRES_DB=velour -e POSTGRES_USERNAME=postgres -e POSTGRES_PORT=5432 --network=host migrations

functional-tests:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost POSTGRES_DB=velour POSTGRES_USERNAME=postgres POSTGRES_PORT=5432  pytest -v ./api/tests/functional-tests

start-server:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost uvicorn velour_api.main:app --host 0.0.0.0

integration-tests:
	python -m pytest -v ./integration_tests/client
