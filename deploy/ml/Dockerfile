FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS dependencies
WORKDIR /app/ml
RUN python -m pip install --no-cache-dir uv==0.13.7
COPY ml/pyproject.toml ml/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

FROM golang:1.26.11@sha256:957952ae74dc010608717144399a59a2c6e82ea03d63be1c7ff6577c5a135f93 AS projector
WORKDIR /src
COPY go.mod go.sum ./
COPY internal/incidents/ internal/incidents/
COPY benchmarks/projection-driver/ benchmarks/projection-driver/
RUN CGO_ENABLED=0 go build -trimpath -o /out/netwatch-projection-driver ./benchmarks/projection-driver

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/ml PATH=/app/ml/.venv/bin:$PATH
WORKDIR /app
COPY --from=dependencies /app/ml/.venv/ /app/ml/.venv/
COPY contracts/ contracts/
COPY ml/netwatch_ml/ ml/netwatch_ml/

FROM runtime AS trainer
ENV PYTHONPATH=/app/ml:/app
COPY benchmarks/event_evaluator.py benchmarks/event_evaluator.py
COPY --from=projector /out/netwatch-projection-driver /usr/local/bin/netwatch-projection-driver
USER 65532:65532

FROM runtime AS serving
USER 65532:65532
CMD ["python", "-m", "netwatch_ml.serving", "--host", "0.0.0.0", "--port", "8000"]
