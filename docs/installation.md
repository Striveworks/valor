# Installation

Valor is comprised of two services: a back end service (which consists of a REST API and a Postgres DB with the PostGIS extension), and a python client for interacting with the back end service.

## Setting up the back end service

### Using docker compose

The easiest way to get up and running with Valor is to use docker compose with the `docker-compose.yml` file in the repo root:

```shell
git clone https://github.com/striveworks/valor
cd valor
docker compose --env-file ./api/.env.testing up
```

This will set up the necessary environment variables, start both the API and database services, and run the database migration job. The endpoint `localhost:8000/health` should return `{"status":"ok"}` if all of Valor's services were started correctly.

**Note: running Valor this way is not intended for production and scalable use, and is only recommended for development and testing purposes**.

### Deploying via Docker and a hosted database

For a more production grade deployment, we publish the images `ghcr.io/striveworks/valor/valor-service` (used for the REST API) and `ghcr.io/striveworks/valor/migrations` (used for setting up the database and migrations). These can be paired with any Postgres database with the PostGIS extension.

The following environment variables are required for running these images:

| Variable | Description | Images that need it |
| --- | --- | --- |
| `POSTGRES_HOST` | The host of the Postgres database | `valor-service`, `migrations` |
| `POSTGRES_PORT` | The port of the Postgres database | `valor-service`, `migrations` |
| `POSTGRES_DB` | The name of the Postgres database | `valor-service`, `migrations` |
| `POSTGRES_USERNAME` | The user of the Postgres database | `valor-service`, `migrations` |
| `POSTGRES_PASSWORD` | The password of the Postgres database | `valor-service`, `migrations` |
| `POSTGRES_SSLMODE` | Sets the Postgres instance SSL mode (typically needs to be "require") | `migrations` |
| `API_ROOT_PATH` | The root path of the API (if serving behind a proxy) | `valor-service` |

Additionally, the Valor REST API has an optional single username/password/bearer token authentication. To enable this feature, the `valor-service` image requires the following environment variables:

| Variable | Description |
| --- | --- |
| `VALOR_USERNAME` | The username to use |
| `VALOR_PASSWORD` | The password to use |
| `VALOR_SECRET_KEY` | A random, secret string used for signing JWT tokens |

### Helm Chart

Alternatively, we provide pre-built charts for deplying Valor on Kubernetes via Helm using the following commands:

```shell
helm repo add valor https://striveworks.github.io/valor-charts/
helm install valor valor/valor
# Valor should now be avaiable at valor.namespace.svc.local
```

### Manual Deployment

If you would prefer to build your own image or want a debug console for the back end, please see the deployment instructions in [Contributing to Valor](contributing.md).

## Setting up the Python client

The python client can be installed via pip:

```shell
pip install valor-client
```
