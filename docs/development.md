### 2. Setup pre-commit

To ensure formatting consistency, we use [pre-commit](https://pre-commit.com/) to manage git hooks. To install pre-commit, run:

```bash
pip install pre-commit
pre-commit install
```


```bash

python -m pip install -e client/.[test]

```


## Dev setup

### 1. Install Docker

As a first step, be sure your machine has Docker installed. [Click here](https://docs.docker.com/engine/install/) for basic installation instructions.

### 2. Setup pre-commit

To ensure formatting consistency, we use [pre-commit](https://pre-commit.com/) to manage git hooks. To install pre-commit, run:

```bash
pip install pre-commit
pre-commit install
```

### 3. Install the client module

Run the following in bash (note: may not work in zsh):

```bash

python -m pip install -e client/.[test]

```




### 4. Run the API, postgis, and redis services

There are two ways to develop locally:
1. *Run everything in Docker*: This method is easier to setup and mirrors how velour will be used in production, but prevents you from debugging `api/*` directly.
2. *Run postgis and redis in Docker, but run the API service locally*: Slightly more difficult to setup, but allows you to debug `api/*` as you code.

#### Approach #1: Run everything in Docker

Simply run:

```shell

make dev-env

```

#### Approach #2: Run the API service locally

Make and/or activate a python 3.10+ environment:

```shell

conda create --name velour_api_env python=3.11
conda activate velour_api_env

```

Install the `api` directory in bash (note: may not work in zsh):

```bash

python -m pip install -e api/.[test]

```

Start the postgis and redis containers:

```shell

make start-redis-docker
make start-postgis-docker

```

Start the service:

```shell

make start-server

```

### 5. (Optional) Setup pgAdmin to debug postgis

You can use the pgAdmin utility to debug your postgis tables as you code. Start by [installing pgAdmin](https://www.pgadmin.org/download/), then select `Object > Register > Server` to connect to your postgis container:
- *Host name/address*: 0.0.0.0
- *Port*: 5432
- *Maintenance database*: postgres
- *Username*: postgres


### 6. Try it out!
We'd recommend starting with the notebooks in `sample_notebooks/*.ipynb`.


## Release process

A release is made by publishing a tag of the form `vX.Y.Z` (e.g. `v0.1.0`). This will trigger a GitHub action that will build and publish the python client to [PyPI](https://pypi.org/project/velour-client/). These releases should be created using the [GitHub UI](https://github.com/Striveworks/velour/releases).

## Tests

There are integration tests, backend unit tests, and backend functional tests.

### CI/CD

All tests are run via GitHub actions on every push.

### Running locally

These can be run locally as follows:

#### Integration tests

1. Install the client: from the `client` directory run

```shell
pip install ".[test]"
```

2. Install the backend: from the `api` directory run

```shell
pip install .[test]
```

3. Setup the backend test env (which requires docker compose): from the `api` directory run

```shell
make test-env
```

4. Run the tests: from the base directory run

```shell
pytest -v integration_tests
```

#### Backend unit tests

1. Install the backend package: from the `api` directory run

```shell
pip install .[test]
```

2. Run the tests: from the `api` directory run

```shell
pytest -v tests/unit-tests
```

#### Backend functional tests

These are tests of the backend that require a running instance of PostGIS to be running. To run these

1. Install the backend package: from the `api` directory run

```shell
pip install .[test]
```

1. Set the environment variables `POSTGRES_HOST` (e.g., `export POSTGRES_HOST=0.0.0.0`) and `POSTGRES_PASSWORD` (e.g., `export POSTGRES_PASSWORD=password`) to a running PostGIS instance.

2. Run the functional tests: from the `api` directory run

```shell
pytest -v tests/functional-tests/test_client.py
```

## Authentication

The API can be run without authentication (by default) or with authentication provided by [auth0](https://auth0.com/). A small react app (code at `web/`)

### Backend

To enable authentication for the backend either set the environment variables `AUTH_DOMAIN`, `AUTH_AUDIENCE`, and `AUTH_ALGORITHMS` or put them in a file named `.env.auth` in the `api` directory. An example of such a file is

```
AUTH0_DOMAIN="velour.us.auth0.com"
AUTH0_AUDIENCE="https://velour.striveworks.us/"
AUTH0_ALGORITHMS="RS256"
```

### Frontend

For the web UI either set the environment variables `VITE_AUTH0_DOMAIN`, `VITE_AUTH0_CLIENT_ID`, `VITE_AUTH0_CALLBACK_URL` and `VITE_AUTH0_AUDIENCE` or put them in a file named `.env` in the `web` directory. The URL to the should also be provided under the `VITE_BACKEND_URL`.  An example of such a file is:

```
VITE_AUTH0_DOMAIN=velour.us.auth0.com
VITE_AUTH0_CLIENT_ID=<AUTH0 CLIENT ID>
VITE_AUTH0_CALLBACK_URL=http://localhost:3000/callback
VITE_AUTH0_AUDIENCE=https://velour.striveworks.us/
VITE_BACKEND_URL=http://localhost:8000
```

Currently, the Striveworks UI Library, Minerva, is closed-source.  This requires access to the Striveworks Repo and an access token.
In order start the web container for Velour, a `.npmrc` file is required in the `web` directory.  Replace `GITHUB AUTH TOKEN` in the following example file:

```
@striveworks:registry = "https://npm.pkg.github.com/"
//npm.pkg.github.com/:_authToken = <GITHUB AUTH TOKEN>
```

### Testing auth

All tests mentioned above run without authentication except for `integration_tests/test_client_auth.py`. Running this test requires setting the envionment variables `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_CLIENT_ID`, and `AUTH0_CLIENT_SECRET` accordingly.

## Deployment settings

For deploying behind a proxy or with external routing, the environment variable `API_ROOT_PATH` can be set in the backend, which sets the `root_path` arguement to `fastapi.FastAPI` (see https://fastapi.tiangolo.com/advanced/behind-a-proxy/#setting-the-root_path-in-the-fastapi-app)
