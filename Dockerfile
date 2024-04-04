ARG PYTHON_VERSION=3.11.4

FROM python:${PYTHON_VERSION} as base

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN mkdir -p /app/.models
RUN mkdir -p /app/.models/data

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

COPY . .

RUN python ml_train_pipeline.py

VOLUME /app/.models
VOLUME /app/.models/data

EXPOSE 8000

CMD uvicorn 'main:app' --host=0.0.0.0 --port=8000
