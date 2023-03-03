# velour evaluation store

![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ekorman/501428c92df8d0de6805f40fb78b1363/raw/velour-coverage.json)

This repo contains the python [client](client) and [backend](backend) packages.

## Tests

There are integration tests, backend unit tests, and backend functional tests.

### CI/CD

All tests are run via GitHub actions on every push.

### Running locally

These can be run locally as follows:

#### Integration tests

1. Install the client: from the `client` directory run

```shell
pip install .[test]
```

2. Install the backend: from the `backend` directory run

```shell
pip install .[test]
```

3. Setup the backend test env (which requires docker compose): from the `backend` directory run

```shell
make test-env
```

4. Run the tests: from the base directory run

```shell
pytest -v integration_tests
```

#### Backend unit tests

1. Install the backend package: from the `backend` directory run

```shell
pip install .[test]
```

2. Run the tests: from the `backend` directory run

```shell
pytest -v tests/unit-tests
```

#### Backend functional tests

These are tests of the backend that require a running instance of PostGIS to be running. To run these

1. Install the backend package: from the `backend` directory run

```shell
pip install .[test]
```

2. Set the environment variaables `POSTGRES_HOST` and `POSTGRES_PASSWORD` to a running PostGIS instance.

3. Run the functional tests: from the `backend` directory run

```shell
pytest -v tests/functional-tests
```
