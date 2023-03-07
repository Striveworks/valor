FROM python:3.10-slim

RUN apt-get update
RUN apt-get install build-essential libpq-dev -y

COPY ./pyproject.toml /src/
COPY ./velour_api /src/velour_api

WORKDIR /src
RUN python -m pip install -U pip
RUN python -m pip install .

ENV API_ROOT_PATH /

CMD "uvicorn velour_api.main:app --reload --root-path ${API_ROOT_PATH} --host 0.0.0.0"