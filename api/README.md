# valor back end

## docker compose envs

To bring up a testing environment, which runs the current back end code and a postgres instance with the PostGIS extension, run

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

This will mount the back end code into the container and live refresh the back end service upon code updates.
