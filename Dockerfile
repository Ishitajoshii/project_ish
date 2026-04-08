FROM node:22-slim AS frontend-build

WORKDIR /app/ui

COPY ui/package.json ui/package-lock.json ./
RUN npm ci

COPY ui ./
RUN npm run build


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:/home/user/.local/bin:$PATH

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR $HOME/app

COPY --chown=user pyproject.toml ./
COPY --chown=user uv.lock ./
RUN uv export \
        --frozen \
        --format requirements.txt \
        --no-dev \
        --no-editable \
        --no-emit-project \
        --prune openenv-core \
        --output-file runtime-requirements.txt && \
    uv venv && \
    uv pip install --python .venv/bin/python -r runtime-requirements.txt && \
    rm runtime-requirements.txt

COPY --chown=user openenv.yaml ./
COPY --chown=user models.py ./
COPY --chown=user client.py ./
COPY --chown=user inference.py ./
COPY --chown=user server ./server
COPY --chown=user tasks ./tasks
COPY --chown=user --from=frontend-build /app/ui/dist ./ui/dist

RUN uv pip install --python .venv/bin/python --no-deps .

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
