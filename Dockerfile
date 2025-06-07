FROM snakepacker/python:3.10 AS builder

RUN apt-get update && apt install -y python3.10-venv
RUN python3.10 -m venv /app && /app/bin/pip install -U pip


COPY requirements.txt /requirements.txt
RUN /app/bin/python3.10 -m pip install --no-cache-dir -r /requirements.txt


############################

FROM snakepacker/python:3.10 AS api

COPY --from=builder /app /app

COPY ./src /app/src
COPY ./configs /app/src/configs
COPY ./weights /app/src/weights
COPY ./embeddings_dict.pickle /app/src/embeddings_dict.pickle

CMD ["/app/bin/gunicorn", "--workers=1", "--bind= 0.0.0.0:8080", "-k=uvicorn.workers.UvicornWorker", "--chdir=/app/src",\
    "--timeout=300", "--log-level=debug", "--keep-alive=5","--preload", "app:app"]
