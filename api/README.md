# valor backend

## docker compose envs

To bring up a testing environment, which runs the current backend code and a postgres instance with the PostGIS extension, run

```shell
make test-env
```

and bring down with

```shell
make stop-env
```

For local development you can run

```shell
make dev-env
```

This will mount the backend code into the container and live refresh the backend service upon code updates.
