FROM python:3.10-slim

ARG VERSION="0.0.0-dev"

RUN apt-get update
RUN apt-get install build-essential libpq-dev -y

COPY ./pyproject.toml /src/
COPY ./velour_api /src/velour_api

WORKDIR /src
RUN python -m pip install -U pip
# when building, we'll use setuptools_scm on the hostmachine to get the version
# and then pass it to the build arg VERSION. this is so we don't need to install
# git and put .git (which setuptools_scm needs to determine the version) in the container
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} python -m pip install .

CMD ["uvicorn", "velour_api.main:app", "--reload", "--host", "0.0.0.0"]
