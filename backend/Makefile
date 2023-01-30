test-env:
	docker compose -p velour --env-file .env.testing up --build -d

dev-env:
	docker compose -p velour -f docker-compose.yml -f docker-compose.dev.yml --env-file .env.testing up --build

stop-env:
	docker compose -p velour down

unit-tests:
	python -m pytest -v tests/unit-tests