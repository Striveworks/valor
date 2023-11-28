# Backend

The backend consists of three components

1. A `velour` REST API service.
2. A PostgreSQL instance with the PostGIS extention.
3. A redis instance


## Helm (Recommended)

## Docker

An image for the backend REST API service is hosted on GitHub's Container registry at `ghcr.io/striveworks/velour/velour-service`. Until the velour repo becomes public, you will need to authenticate to pull the image. To do this, you need to create a personal access token here https://github.com/settings/tokens that has read access to GitHub packages. Then run

```shell
docker login ghcr.io
```

and enter your username and the access token as the password.

## Docker Compose

The Docker compose file [here](https://github.com/Striveworks/velour/blob/main/backend/docker-compose.yml) sets up all three services with the appropriate networking. To run, set the environment variable `POSTGRES_PASSWORD` to your liking and then run

```shell
docker compose up
```
