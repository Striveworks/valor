test-env:
	docker compose -p valor --env-file ./api/.env.testing up --build -d

dev-env:
	docker compose -p valor -f docker-compose.yml --env-file ./api/.env.testing up --build -d

stop-env:
	docker compose -p valor down

unit-tests:
	python -m pytest -v ./api/tests/unit-tests
	python -m pytest -v ./client/unit-tests

start-postgis-docker:
	docker run -p 5432:5432 -e POSTGRES_PASSWORD=password -e POSTGRES_DB=valor -d docker.io/postgis/postgis

run-migrations:
	docker build -f=migrations/Dockerfile ./migrations -t migrations && \
	docker run -e POSTGRES_PASSWORD=password -e POSTGRES_HOST=host.docker.internal -e POSTGRES_DB=valor -e POSTGRES_USERNAME=postgres -e POSTGRES_PORT=5432 migrations

functional-tests:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost POSTGRES_DB=valor POSTGRES_USERNAME=postgres POSTGRES_PORT=5432  pytest -v ./api/tests/functional-tests

start-server:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost POSTGRES_DB=valor uvicorn valor_api.main:app --host 0.0.0.0

integration-tests:
	python -m pytest -v ./integration_tests/client