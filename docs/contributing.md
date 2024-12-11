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

Install the `valor-lite` module along with any packages required for development:
```bash
make install-dev
```

### Running Tests

All of our tests are run automatically via GitHub Actions on every push, so it's important to double-check that your code passes all local tests before committing your code.

```shell
make pre-commit
make tests
make external-tests
```
