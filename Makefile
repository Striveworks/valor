test-env:
	docker compose -p valor --env-file ./api/.env.testing up --build -d

dev-env:
	docker compose -p valor -f docker-compose.yml --env-file ./api/.env.testing up --build -d

stop-env:
	docker compose -p valor down

unit-tests:
	python -m pytest -v ./api/tests/unit-tests
	python -m pytest -v ./client/unit-tests

start-postgres-docker:
	docker build -t pgvalor ./database
	docker run -p 5432:5432 -e POSTGRES_PASSWORD=password -e POSTGRES_DB=valor -d pgvalor

start-constrained-postgres-docker:
	docker build -t pgvalor ./database
	docker run \
		--cpus="1" \
		--memory 4G \
		--memory-swap -1 \
		-p 5432:5432 \
		-e POSTGRES_PASSWORD=password \
		-e POSTGRES_DB=valor -d pgvalor

		# --device-read-bps /dev/sdb:20mb \
		# --device-write-bps /dev/sdb:40mb \


run-migrations:
ifeq ($(shell uname -s),Darwin)
	docker build -f=migrations/Dockerfile ./migrations -t migrations && \
	docker run -e POSTGRES_PASSWORD=password -e POSTGRES_HOST=host.docker.internal -e POSTGRES_DB=valor -e POSTGRES_USERNAME=postgres -e POSTGRES_PORT=5432 migrations
else
	docker build -f=migrations/Dockerfile ./migrations -t migrations && \
	docker run -e POSTGRES_PASSWORD=password --network "host" -e POSTGRES_HOST=localhost -e POSTGRES_DB=valor -e POSTGRES_USERNAME=postgres -e POSTGRES_PORT=5432 migrations
endif

functional-tests:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost POSTGRES_DB=valor POSTGRES_USERNAME=postgres POSTGRES_PORT=5432  pytest ./api/tests/functional-tests

start-server:
	POSTGRES_PASSWORD=password POSTGRES_HOST=localhost POSTGRES_DB=valor uvicorn valor_api.main:app --host 0.0.0.0

integration-tests:
	python -m pytest -v ./integration_tests/client
