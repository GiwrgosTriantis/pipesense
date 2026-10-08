# syntax=docker/dockerfile:1.7
# One image for every stage, every environment and every orchestrator:
#   docker run pipesense:<git-sha> run score --utility fairport --job-id job-1
# Targets:  runtime (default, what ships)  |  test (runtime + dev deps + tests; runs pytest)
#
# Base images are pinned by digest: the tag is for humans, the digest is what gets built.
# Refresh both on purpose, for example to pick up OS security patches:
#   docker buildx imagetools inspect python:3.12.15-slim-bookworm
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ARG UV_IMAGE=astral/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c

FROM ${UV_IMAGE} AS uv

# ---- builder: resolve nothing, install exactly what uv.lock says ----
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first: this layer stays cached until uv.lock changes, not on every code change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM builder AS builder-test
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

# ---- runtime: slim base, venv only, no build tools, no data, no secrets, non-root ----
FROM ${PYTHON_IMAGE} AS runtime
ARG GIT_SHA=unknown
ARG IMAGE_TAG=dev
LABEL org.opencontainers.image.title="pipesense" \
      org.opencontainers.image.description="PipeSense leak-risk batch pipeline" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${IMAGE_TAG}"
RUN groupadd --system --gid 10001 pipesense \
 && useradd --system --uid 10001 --gid pipesense --home-dir /app --shell /usr/sbin/nologin pipesense \
 && mkdir -p /data/runs && chown -R 10001:10001 /data   # dev profile mounts a volume here
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY config/pipeline.toml /app/config/pipeline.toml
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PIPESENSE_PIPELINE_CONFIG=/app/config/pipeline.toml \
    PIPESENSE_GIT_SHA=${GIT_SHA} \
    PIPESENSE_IMAGE_TAG=${IMAGE_TAG}
USER 10001:10001
ENTRYPOINT ["pipesense"]
CMD ["--help"]

# ---- test: the same runtime image plus dev deps and tests ----
FROM runtime AS test
COPY --from=builder-test /opt/venv /opt/venv
COPY pyproject.toml /app/pyproject.toml
COPY tests /app/tests
COPY data/raw /app/data/raw
ENTRYPOINT ["python", "-m", "pytest", "-p", "no:cacheprovider"]
CMD ["-q"]
