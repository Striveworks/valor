test-env:
	docker-compose -p velour --env-file .env.testing up --build -d

dev-env:
<<<<<<< HEAD
	docker-compose -p velour -f docker-compose.yml -f docker-compose.dev.yml --env-file .env.testing up --build
=======
	docker compose -p velour -f docker compose.yml -f docker compose.dev.yml --env-file .env.testing up --build
>>>>>>> 76-chunk-data-transfer-from-client-to-backend

stop-env:
	docker-compose -p velour down

unit-tests:
	python -m pytest -v tests/unit-tests

start-postgis-docker:
	docker run -p 5432:5432 -e POSTGRES_PASSWORD=password -d postgis/postgis

start-redis-docker:
	docker run -p 6379:6379 -d redis

functional-tests:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost pytest -v tests/functional-tests

start-server:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost uvicorn velour_api.main:app --reload --host 0.0.0.0
