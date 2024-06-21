FROM python:3.10-slim

ARG VALOR_VERSION="0.0.0-dev"

RUN apt-get update && apt-get install build-essential libpq-dev -y

COPY ./pyproject.toml /src/

WORKDIR /src
RUN python -m pip install -U pip
# when building, we'll use setuptools_scm on the hostmachine to get the version
# and then pass it to the build arg VALOR_VERSION. this is so we don't need to install
# git and put .git (which setuptools_scm needs to determine the version) in the container
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VALOR_VERSION} python -m pip install .
COPY ./valor_api /src/valor_api

CMD ["uvicorn", "valor_api.main:app", "--host", "0.0.0.0", "--log-level", "warning"]
