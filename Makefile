test-env:
	docker-compose -p velour --env-file ./api/.env.testing up --build -d

dev-env:
	docker-compose -p velour -f docker-compose.yml -f docker-compose.dev.yml --env-file ./api/.env.testing up --build

stop-env:
	docker-compose -p velour down

unit-tests:
	python -m pytest -v ./api/tests/unit-tests

start-postgis-docker:
	docker run -p 5432:5432 -e POSTGRES_PASSWORD=password -d postgis/postgis

start-redis-docker:
	docker run -p 6379:6379 -e ALLOW_EMPTY_PASSWORD=yes -d public.ecr.aws/bitnami/redis:7.0

functional-tests:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost pytest -v ./api/tests/functional-tests

start-server:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost uvicorn velour_api.main:app --host 0.0.0.0
