# Minimal runtime image for the FastAPI/OpenEnv server.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY uv.lock ./
COPY openenv.yaml ./
COPY models.py ./
COPY client.py ./
COPY inference.py ./
COPY server ./server
COPY tasks ./tasks

RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
