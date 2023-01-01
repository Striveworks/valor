FROM python:3.10-slim

RUN apt-get update
RUN apt-get install build-essential libpq-dev -y

COPY ./pyproject.toml /

RUN python -m pip install .

COPY ./velour_api /velour_api
WORKDIR /

CMD ["uvicorn", "velour_api.main:app", "--reload", "--host", "0.0.0.0"]