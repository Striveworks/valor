FROM python:3.10-slim

RUN apt-get update
RUN apt-get install build-essential libpq-dev -y

COPY ./requirements.txt /requirements.txt

RUN python -m pip install -r requirements.txt

COPY ./velour_api /velour_api
WORKDIR /

CMD ["uvicorn", "velour_api.main:app", "--reload", "--host", "0.0.0.0"]