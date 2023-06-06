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
pytest -v tests/functional-tests/test_client.py
```

## Authentication

The API can be run without authentication (by default) or with authentication provided by [auth0](https://auth0.com/). A small react app (code at `web/`)

### Backend

To enable authentication for the backend either set the environment variables `AUTH_DOMAIN`, `AUTH_AUDIENCE`, and `AUTH_ALGORITHMS` or put them in a file named `.env.auth` in the `backend` directory. An example of such a file is

```
AUTH0_DOMAIN="velour.us.auth0.com"
AUTH0_AUDIENCE=***REMOVED***
AUTH0_ALGORITHMS="RS256"
```

### Frontend

For the web UI either set the environment variables `VITE_AUTH0_DOMAIN`, `VITE_AUTH0_CLIENT_ID`, `VITE_AUTH0_CALLBACK_URL` and `VITE_AUTH0_AUDIENCE` or put them in a file named `.env` in the `web` directory. An example of such a file is:

```
VITE_AUTH0_DOMAIN=velour.us.auth0.com
VITE_AUTH0_CLIENT_ID=JHsL3WgCueWyKi0mnBl8o47r0Ux6XG8P
VITE_AUTH0_CALLBACK_URL=http://localhost:3000/callback
VITE_AUTH0_AUDIENCE=https://velour.striveworks.us/
```

### Testing auth

All tests mentioned above run without authentication except for `integration_tests/test_client_auth.py`. Running this test requires setting the envionment variables `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_CLIENT_ID`, and `AUTH0_CLIENT_SECRET` accordingly.

## Deployment settings

For deploying behind a proxy or with external routing, the environment variable `API_ROOT_PATH` can be set in the backend, which sets the `root_path` arguement to `fastapi.FastAPI` (see https://fastapi.tiangolo.com/advanced/behind-a-proxy/#setting-the-root_path-in-the-fastapi-app)
