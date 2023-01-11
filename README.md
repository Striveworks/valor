# velour

This repo contains the python [client](client) and [backend](backend) packages.

## Tests

There are integration tests, client unit tests, and backend unit tests.

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

#### Client unit tests

1. Install the client: from the `client` directory run

```shell
pip install .[test]
```

2. Run the tests: from the `client` directory run

```shel
pytest -v tests
```

#### Backend unit tests

1. Install the client: from the `backend` directory run

```shell
pip install .[test]
```

2. Run the tests: from the `backend` directory run

```shel
pytest -v tests
```
