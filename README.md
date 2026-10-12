# PipeSense: leak-risk pipeline

Batch pipeline that ranks a water utility's pipes by leak risk. It is containerised, reproducible and
resumable, with a gated MLflow model lifecycle and a queue-driven miniature of the AWS design.

- **DECISIONS.md:** review findings, assumptions, decisions and their alternatives, ML results, known debt, AI usage
- **DESIGN.md:** how this runs on SageMaker (AWS) for hundreds of utilities
- **ASSESSMENT.md:** the brief

## Guarantees

Each row is proved by the named tests (`tests/<file>.py::<test>`) and, where a command is given, shown by the
walkthrough below.

| Guarantee | Proved by | Demonstrated by |
|---|---|---|
| A completed stage does not run again | `test_pipeline::test_resume_after_failure_skips_finished_stages`, `test_worker::test_transient_fault_is_retried_and_resumes_after_finished_stages` | step 2: the second `run score … --job-id job-1` skips both stages |
| A changed raw export reruns `prepare` and everything after it | `test_pipeline::test_changed_raw_data_invalidates_prepare_and_downstream` | — |
| A failed stage is never treated as done: it has no marker and runs again; local writes are atomic | `test_pipeline::test_injected_transient_fault_then_retry_resumes`, `test_storage::test_local_write_leaves_no_temp_files` | step 3: `submit … --inject-fault predict` |
| No pipe is scored by a model that trained on it | `test_model::test_no_pipe_is_scored_by_a_model_that_trained_on_it`, `test_pipeline::test_score_job_never_uses_a_model_that_trained_on_the_pipe` | step 2: `score_source` column in `predictions.csv` |
| A candidate that does not beat the age rule and the champion is rejected, and the reason is recorded | `test_gate::test_first_model_must_beat_the_age_rule`, `test_gate::test_tie_with_champion_is_rejected_no_churn`, `test_pipeline::test_retraining_unchanged_data_does_not_replace_the_champion`, `test_pipeline::test_train_registers_a_version_and_the_gate_decides_with_an_audit_trail` | step 2: `run train` twice; v2 is `REJECTED` |
| A bad export or wrong units fail fast, without retries | `test_worker::test_bad_data_fails_fast_to_the_dlq_without_retry`, `test_pipeline::test_export_in_inches_is_refused_by_prepare_using_the_configured_bounds` | step 3: `submit --utility brokenville`, then `dlq` |
| A transient failure retries and resumes after the finished stages | `test_worker::test_transient_fault_is_retried_and_resumes_after_finished_stages` | step 3: `submit … --inject-fault predict` |
| A worker killed mid-job is detected; the job resumes, or is dead-lettered after the attempt limit | `test_worker::test_crashed_attempt_is_recorded_and_the_next_attempt_resumes`, `test_worker::test_crashes_on_every_attempt_dead_letter_the_job_instead_of_leaving_it_running` | by hand: `docker kill` the worker mid-job (DECISIONS.md section 7) |
| Every prediction row names its model version | `test_pipeline::test_predictions_record_model_version_and_lineage` | step 2: `head … predictions.csv` |

Everything below needs only **Docker with Compose v2**. No AWS account, Python or conda is required.

## 1. Run the tests inside the image

```bash
docker compose --profile test build test
docker compose --profile test run --rm test          # 79 passed, 1 skipped (needs a live Postgres)
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
champion). Every decision is a run in the `leak-risk/fairport/gate` experiment in MLflow, and its report breaks
AP and precision@10% down per material.

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
analysis/           reproduces the code-review numbers (original starter kit) and the num_breaks_5yr ablation
```

Clean up everything: `docker compose --profile dev --profile prod --profile test down -v`.

Local development without Docker (optional): `uv sync` then `uv run pytest`.
