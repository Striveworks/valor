# Contributing to Valor

We welcome all contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas aimed at improving Valor. This doc describes the high-level process for how to contribute to this repository. If you have any questions or comments about this process, please feel free to reach out to us on [Slack](https://striveworks-public.slack.com/join/shared_invite/zt-1a0jx768y-2J1fffN~b4fXYM8GecvOhA#/shared-invite/email).

## On GitHub

We use [Git](https://git-scm.com/doc) on [GitHub](https://github.com) to manage this repo, which means you will need to sign up for a free GitHub account to submit issues, ideas, and pull requests. We use Git for version control to allow contributors from all over the world to work together on this project.

If you are new to Git, these official resources can help bring you up to speed:

- [GitHub documentation for forking a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
- [GitHub documentation for collaborating with pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)
- [GitHub documentation for working with forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks)

## Contribution Workflow

Generally, the high-level workflow for contributing to this repo includes:

1. Submitting an issue or enhancement request using the appropriate template on [GitHub Issues](https://github.com/Striveworks/valor/issues).
2. Gathering feedback from devs and the broader community in your issue _before_ starting to code.
3. Forking the Valor repo, making your proposed changes, and submitting a pull request (PR). When submitting a PR, please be sure to:
     1. Update the README.md and/or any relevant docstrings with details of your change.
     2. Add tests where necessary.
     3. Run `pre-commit install` on your local repo before your last commit to ensure your changes follow our formatting guidelines.
     4. Double-check that your code passes all of the tests that are automated via GitHub Actions.
     5. Ping us on [Slack](https://striveworks-public.slack.com/join/shared_invite/zt-1a0jx768y-2J1fffN~b4fXYM8GecvOhA#/shared-invite/email) to ensure timely review.
4. Working with repo maintainers to review and improve your PR before it is merged into the official repo.


For questions or comments on this process, please reach out to us at any time on [Slack](https://striveworks-public.slack.com/join/shared_invite/zt-1a0jx768y-2J1fffN~b4fXYM8GecvOhA#/shared-invite/email).


## Development Tips and Tricks

### Deploying the Back End for Development

#### Docker Compose

The fastest way to test the API and Python client is via Docker Compose. Start by setting the environment variable `POSTGRES_PASSWORD` to your liking, and then start Docker and build the container:

```shell
export POSTGRES_PASSWORD="my_password"
docker compose up
```

#### Makefile (requires Docker)

Alternatively, you may want to run the API service from a terminal to enable faster debugging. To start the service, you can run:

```shell
pip install api # Install the API in your python environment

export POSTGRES_PASSWORD=password
export POSTGRES_HOST=localhost
make start-postgis-docker # Start the postgis service in Docker
make run-migrations # Instantiate the table schemas in Postgres
make start-server # Start the API service locally
```

### Setting Up Your Environment

Creating a Valor-specific Python environment at the start of development can help you avoid dependency and versioning issues later on. To start, we'd recommend activating a new Python environment:

```bash
# venv
python3 -m venv .env-valor
source .env-valor/bin/activate

# conda
conda create --name valor python=3.11
conda activate valor
```

Next, install [pre-commit](https://pre-commit.com/) to ensure formatting consistency throughout your repo:

```bash
pip install pre-commit
pre-commit install
```

Finally, you're ready to install your client and API modules:

```bash
# Install the Client module
python -m pip install -e client/.

# Install the API module
python -m pip install -e api/.
```

### Use pgAdmin to Debug PostGIS

You can use the pgAdmin utility to debug your PostGIS tables as you code. Start by [installing pgAdmin](https://www.pgadmin.org/download/), and then select `Object > Register > Server` to connect to your PostGIS container. The default connection details are listed below for convenience:

```
- *Host name/address*: 0.0.0.0
- *Port*: 5432
- *Maintenance database*: postgres
- *Username*: postgres
```

### Running Tests

All of our tests are run automatically via GitHub Actions on every push, so it's important to double-check that your code passes all local tests before committing your code. All of the tests below require `pytest`:

```shell
pip install pytest
```


#### Running integration tests

```shell
pytest integration_tests
```

#### Running back end unit tests

```shell
pytest api/tests/unit-tests
```

#### Running back end functional tests

> **Note:** Functional tests require a running instance of PostgreSQL, which you can start using `make start-postgis-docker`.

```shell
POSTGRES_PASSWORD=password \
POSTGRES_HOST=localhost \
pytest api/tests/functional-tests/
```
