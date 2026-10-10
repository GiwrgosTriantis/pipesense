# PipeSense: leak-risk pipeline

Batch pipeline that ranks a water utility's pipes by leak risk. It is containerised, reproducible and
resumable, with a gated MLflow model lifecycle and a queue-driven miniature of the AWS design.

- **DECISIONS.md:** review findings, assumptions, decisions and their alternatives, ML results, known debt, AI usage
- **DESIGN.md:** how this runs on AWS for hundreds of utilities
- **ASSESSMENT.md:** the brief

Everything below needs only **Docker with Compose v2**. No AWS account, Python or conda is required.

## 1. Run the tests inside the image

```bash
docker compose --profile test build test
docker compose --profile test run --rm test          # 63 passed, 1 skipped (needs a live Postgres)
```

## 2. Development setup (local volume + MLflow on sqlite)

```bash
docker compose --profile dev build                                    # builds pipesense:dev
docker compose --profile dev up -d mlflow-dev                         # MLflow UI: http://localhost:5000

# train a candidate; the promotion gate decides whether it becomes the champion
docker compose --profile dev run --rm pipeline-dev run train --utility fairport
docker compose --profile dev run --rm pipeline-dev models --utility fairport

# a customer job: score Fairport with the champion
docker compose --profile dev run --rm pipeline-dev run score --utility fairport --job-id job-1
docker compose --profile dev run --rm pipeline-dev run score --utility fairport --job-id job-1   # both stages skipped
docker compose --profile dev run --rm --entrypoint head pipeline-dev -5 /data/runs/fairport/job-1/predict/predictions.csv
```

Training a second time on unchanged data registers v2, and the gate **rejects** it (a tie never replaces the
champion). Every decision is a run in the `leak-risk/fairport/gate` experiment in MLflow.

## 3. Production-like setup + the queue miniature (bonus)

Uses the image built in step 2. Prod services never build it.

```bash
docker compose --profile prod up -d        # postgres, S3 API, MLflow on Postgres, SQS API, worker, status page
docker compose --profile prod run --rm pipeline-prod run train --utility fairport
```

Open the **job status page at http://localhost:8080**. It updates itself. MLflow is at http://localhost:5001.

```bash
# a normal job
docker compose --profile prod run --rm pipeline-prod submit --utility fairport

# a transient failure: predict fails on attempt 1 -> RETRYING -> attempt 2 skips prepare -> SUCCEEDED
docker compose --profile prod run --rm pipeline-prod submit --utility fairport --inject-fault predict

# a permanent failure: export without a material column -> FAILED at once, no retries, sent to the DLQ
docker compose --profile prod run --rm pipeline-prod submit --utility brokenville
docker compose --profile prod run --rm pipeline-prod dlq                # what is dead-lettered, and why

docker compose --profile prod logs -f worker                           # structured JSON logs
```

## 4. How a change travels from dev to prod-like

```bash
export IMAGE_TAG=$(git rev-parse --short HEAD) GIT_SHA=$(git rev-parse --short HEAD)
docker compose --profile test build test && docker compose --profile test run --rm test
docker compose --profile dev build                 # pipesense:<sha>, exercised in dev as in step 2
docker compose --profile prod up -d                # prod-like now runs the same pipesense:<sha>, no rebuild
# rollback a bad image:  IMAGE_TAG=<previous-sha> docker compose --profile prod up -d
# rollback a bad model:  docker compose --profile prod run --rm pipeline-prod promote \
#                          --utility fairport --version <n> --reason "rollback: ..."
```

Only configuration differs between environments: `config/dev.env` vs `config/prod.env`. Pipeline
parameters and gate thresholds are in `config/pipeline.toml` and are baked into the image.

## Layout

```
src/pipesense/
  cli.py            the single entry point (run, stage, promote, models, seed, submit, worker, dlq, status-server)
  runner.py         stage runner: success markers, fingerprints, resume
  stages/           prepare, train, gate, predict: pure functions of a StageContext
  features.py       cleaning + feature engineering (no I/O)
  model.py          model pipeline, spatial holdout, cross-fitted scoring model, metrics, age-rule baseline
  gate.py           promotion rules ("what winning means")
  drift.py          training-data profile and per-feature PSI for scoring jobs
  registry.py       MLflow registry helpers + audit trail
  storage.py        local / S3 artifact store
  worker.py         SQS worker: retries, backoff, DLQ, heartbeat
  status*.py        job status (file or Postgres) + status page
config/             pipeline.toml (what), dev.env / prod.env (where)
deploy/             MLflow server image, Postgres init, SeaweedFS and ElasticMQ config
data/raw/<utility>/ raw exports (pipes.csv + meta.json); brokenville is broken on purpose
analysis/           reproduces the code-review numbers against the original starter kit
```

Clean up everything: `docker compose --profile dev --profile prod --profile test down -v`.

Local development without Docker (optional): `uv sync` then `uv run pytest`.
