#!/usr/bin/env bash

set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

cd "$REPO_DIR"

printf "==> Running pytest\n"
uv run --extra dev pytest

printf "==> Running openenv validate\n"
uv run openenv validate .

printf "==> Building Docker image\n"
docker build -t circuitrl-submission-check .

if [ -n "$PING_URL" ]; then
  PING_URL="${PING_URL%/}"
  printf "==> Pinging Hugging Face Space\n"
  curl -fsS "$PING_URL/health"
  printf "\n==> Probing reset()\n"
  curl -fsS -X POST "$PING_URL/reset" -H "Content-Type: application/json" -d '{}'
  printf "\n"
fi

printf "==> Submission checks passed\n"
