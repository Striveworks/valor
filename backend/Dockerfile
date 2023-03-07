FROM python:3.10-slim

RUN apt-get update
RUN apt-get install build-essential libpq-dev -y

COPY ./pyproject.toml /src/
COPY ./velour_api /src/velour_api

WORKDIR /src
RUN python -m pip install -U pip
RUN python -m pip install .

CMD ["uvicorn", "velour_api.main:app", "--reload", "--host", "0.0.0.0"]