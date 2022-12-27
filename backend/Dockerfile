FROM python:3.10-slim

COPY ./main.py ./main.py
COPY ./requirements.txt ./requirements.txt

RUN python -m pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--reload"]