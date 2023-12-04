# Development

![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ekorman/501428c92df8d0de6805f40fb78b1363/raw/velour-coverage.json)

This repo contains the python [client](client) and [backend api](api) packages for velour. For velour's user documentation, [click here](https://striveworks.github.io/velour/).

# Getting Started

## 1. Install Docker

As a first step, be sure your machine has Docker installed. [Click here](https://docs.docker.com/engine/install/) for basic installation instructions.

## 2. Install Dependencies

Create a Python enviroment using your preferred method.

```bash
# venv
python3 -m venv .env-velour
source .env-velour/bin/activate

# conda
conda create --name velour python=3.11
conda activate velour
```

To ensure formatting consistency, we use [pre-commit](https://pre-commit.com/) to manage git hooks. To install pre-commit, run:

```bash
pip install pre-commit
pre-commit install
```

Install the client module.

```bash
pip install client/[test]
```

# API Development

Velour offers multiple methods of deploying the backend. If you do not require the ability to debug the API, please skip this section and follow the instructions in `Setting up the Backend`.

<details>
<summary>Deploy for development.</summary>

1. Install dependencies.

```bash
# install the api module.
pip install api/[test]
```

2. Launch the containers.

```bash
# launch PostgreSQL in background.
make start-postgis

# launch Redis in background.
make start-redis

# launch the Server.
make start-server
```

</details>

# (Optional) Setup pgAdmin to debug postgis

You can use the pgAdmin utility to debug your postgis tables as you code. Start by [installing pgAdmin](https://www.pgadmin.org/download/), then select `Object > Register > Server` to connect to your postgis container:
- *Host name/address*: 0.0.0.0
- *Port*: 5432
- *Maintenance database*: postgres
- *Username*: postgres

# Try it out!

We'd recommend starting with the notebooks in `sample_notebooks/*.ipynb`.

# Release process

A release is made by publishing a tag of the form `vX.Y.Z` (e.g. `v0.1.0`). This will trigger a GitHub action that will build and publish the python client to [PyPI](https://pypi.org/project/velour-client/). These releases should be created using the [GitHub UI](https://github.com/Striveworks/velour/releases).

## Tests

There are integration tests, backend unit tests, and backend functional tests.

## CI/CD

All tests are run via GitHub actions on every push.

## Running locally

Tests can be run locally using Pytest as follows.

```shell
# install pytest
pip install pytest
```

### Integration tests

```shell
pytest integration_tests
```

### Backend unit tests

```shell
pytest api/tests/unit-tests
```

### Backend functional tests

> **Note:** Functional tests require a running instance of PostgreSQL.

```shell
POSTGRES_PASSWORD=password \
POSTGRES_HOST=localhost \
pytest api/tests/functional-tests/
```