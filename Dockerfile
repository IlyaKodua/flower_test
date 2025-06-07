FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /app && /app/bin/pip install -U pip

COPY requirements.txt /requirements.txt

RUN /app/bin/pip install --no-cache-dir -r /requirements.txt

FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS api

RUN apt-get update && apt-get install -y python3.10 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app /app

COPY ./src /app/src
COPY ./configs /app/src/configs
COPY ./weights /app/src/weights
COPY ./embeddings_dict.pickle /app/src/embeddings_dict.pickle

WORKDIR /app/src

CMD ["/app/bin/gunicorn", "--workers=1", "--bind=0.0.0.0:8080", "-k=uvicorn.workers.UvicornWorker", "--chdir=/app/src", "--timeout=300", "--log-level=debug", "--keep-alive=5", "app:app"]
