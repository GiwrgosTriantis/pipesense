# DECISIONS

Contents: (1) code review findings, (2) assumptions, (3) decisions and the alternatives I weighed,
(4) ML results, (5) known debt, (6) AI usage.

I reviewed the untouched starter kit (first commit) before changing anything. Every finding with a
number was reproduced by running the original code. `analysis/review_checks.py` regenerates the ML
numbers from a run of that commit.

---

## 1. Code review findings

### 1.1 The Dockerfile

| # | Problem | Why it matters |
|---|---------|----------------|
| D1 | `FROM continuumio/miniconda3:latest` | The tag floats, so two builds of the same commit can produce different images. Nothing is pinned by digest. |
| D2 | It installs the whole shared team environment: TensorFlow, Keras, OpenCV, Jupyter, Dask, Streamlit, FAISS and about 70 packages in total | The image runs to several GB, so pulls and cold starts are slow. The attack surface is large, and the CVE scan of a batch job is dominated by notebook servers it never runs. |
| D3 | Runs as root. `groupadd pipesense && usermod -a -G pipesense root` only adds *root* to a new group. | It looks like hardening but does nothing. The container is still root. |
| D4 | `COPY data/pipes.csv /app/data/` bakes customer data into the image | Anyone who can pull the image can read Fairport's network. Every data refresh needs a rebuild, and one image cannot serve many utilities. |
| D5 | `ENV PIPESENSE_ENV=production` and `AWS_DEFAULT_REGION=us-east-1` are baked in | The image declares itself prod, so dev and prod are not "the same image selected by config". Neither variable is read anywhere in the code. |
| D6 | `ENV WORK_DIR=/shared/output` and `DATA_DIR=/app/data` are dead config | `run_all.sh` overwrites both with `./output` and `./data`. Outputs go to the container's ephemeral filesystem and vanish with it unless someone happens to mount `/app/output`. |
| D7 | `CMD ["conda", "run", ...]` | `conda run` captures output by default unless `--no-capture-output` is passed, so logs can be lost or delayed. It also puts a wrapper process between PID 1 and Python, so `docker stop` / SIGTERM does not reach the stage cleanly. |
| D8 | The `SHELL [...]` line is dead | `SHELL` affects only later `RUN` instructions, and there are none. |
| D9 | `tests/` is not copied, and there is no test stage | The tests cannot run inside the image, which the brief requires. |
| D10 | There's no `.dockerignore`, no `conda clean`, no multi-stage build, no labels (git SHA, build date) | Images are bigger than needed and can't be traced back to a commit. |

### 1.2 The environment (`environment.yml`)

- **Only 8 of about 70 packages are used.** The code imports `pandas`, `numpy`, `geopandas` (which pulls in
  `shapely`, `pyproj` and `pyogrio` as the GPKG I/O engine), `scikit-learn` and `matplotlib`. The tests also need `pytest`.
  Everything else is team-wide baggage.
- **It is not reproducible.** Only top-level packages are pinned, so their dependencies resolve fresh on every
  create. Two people who run `conda env create` a month apart get different environments.
- **`prefix: /home/dsteam/anaconda3/envs/pipesense`** is a path on one machine, exported by accident.
- **It mixes the `defaults` and `conda-forge` channels.** This causes solver and ABI conflicts, and the `defaults` channel
  carries commercial licensing terms from Anaconda that a company should check.
- **It installs a pip layer on top of conda with no lock.** It also includes both `psycopg2` and `psycopg` (3), and
  `mlflow-skinny`, which nothing uses yet.

### 1.3 Orchestration and failure handling (`run_all.sh`, `common.py`)

- **F1. A failed run is reported as SUCCESS (reproduced).** `run_all.sh` has no `set -e`. I removed the
  `material` column and ran it on a work dir that held a previous run's output. `prepare_data` failed with an
  `AssertionError`, `train_model` then **retrained on the stale `features.gpkg` from the previous run**,
  `predict` wrote new predictions, `status.json` said `SUCCESS`, and the script exited with **0**. A
  customer gets an analysis of the wrong data, and nobody is alerted.
- **F2. Training failures are never reported (reproduced).** In `train_model.py`, `elapsed` is
  assigned only after `run()` succeeds. On failure, the `except` block raises
  `NameError: name 'elapsed' is not defined`, so `write_status("FAILED")` never runs and the
  real exception is buried under a second one.
- **F3. Concurrent jobs overwrite each other.** All jobs share `WORK_DIR` on NFS and use fixed filenames
  (`features.gpkg`, `model.pkl`, `predictions.csv`, `status.json`). There is no job ID or utility
  in any path. Two customer requests at the same time read each other's files.
- **F4. Writes are not atomic.** A stage that crashes mid-write leaves a truncated file that the next stage (or the
  next run) reads. `status.json` is rewritten in place while a monitor polls it every 30 s, so the
  monitor can read half-written JSON.
- **F5. Status reporting is misleading.** Only `predict` writes `SUCCESS`, and `durations` is a per-process global, so
  the final status shows only the predict stage's duration (reproduced). There is no `RUNNING` state, no
  timestamps, no job ID and no record of which stage failed.
- **F6. A finished stage can't be told apart from a crashed one.** With no completion markers, the only
  "restart" available is to re-run everything.
- **F7. `source ~/.bashrc`** ties the job to whichever user's shell profile is on the VM.
- **F8. `assert` is used for input validation.** Python strips asserts under `-O`, so validation must use real
  exceptions.

### 1.4 Code that won't scale beyond one utility

- **`CURRENT_YEAR = 2026` is hard-coded.** Pipe age should be measured from the date of the data export, not
  from whenever the code was written. The newest `install_year` in the data is 2020, so the export date is unknown.
- **`METRIC_CRS = "EPSG:26915"` (UTM 15N) is hard-coded.** It is correct for Fairport (about 94.8°W), but a utility in
  another UTM zone would get distorted distances and density features. The CRS should be derived per utility,
  for example with `estimate_utm_crs()`.
- **The input path is fixed** at `DATA_DIR/pipes.csv`, and the README says "this copy is set up for Fairport".
  There's no `utility` or `job_id` parameter anywhere.
- **`predict.py` imports `train_model`**, which pulls the training code and matplotlib into the scoring path.
  It also calls `build_matrix`, which **asserts that `has_leak` exists**, so the pipeline cannot score pipes
  that have no label. `build_matrix` also `print`s the whole feature matrix into the logs.
- **Unknown material spellings pass silently.** A spelling missing from `MATERIAL_MAP` (for example `"Cast-Iron"`)
  becomes a new upper-cased category. At scoring time `reindex(fill_value=0)` drops its dummy column, so the
  pipe is scored as if it had **no material**, with no warning. A missing material becomes NaN and crashes
  `impute_missing`. `normalize_materials` also mutates the caller's dataframe.
- **`pickle` is used for the model** with no version, overwritten on every run, and tied to the exact
  scikit-learn version. Unpickling also executes code, so a pickle from an untrusted source is a security risk.
- **Logs** use `print` with ANSI colour codes, which become noise in CloudWatch or any log aggregator.

### 1.5 ML findings

These numbers are on the Fairport data: 3,015 rows, which leaves 2,992 after 15 exact duplicates and 8 rows with
`length_m ≤ 0` are dropped. The leak rate is 15.9%.

- **M1. The pipeline scores its own training data, and the output is mostly known leaks.**
  `predict.py` scores every pipe, including the 75% the model was just trained on. In the
  delivered `predictions.csv`, **all 25 of the top 25 and 96 of the top 100 "riskiest" pipes already have
  `has_leak = 1`**. On that file the average precision (AP) is **0.81**, against **0.52** on held-out data.
  The list a utility receives mostly tells it which pipes it already knows have leaked. The deeper problem is
  how the task is framed: `has_leak` is a historical label, while the utility wants the **risk of a future leak**.
  The target needs a defined observation window, and scoring should cover pipes and time periods the
  model did not see in training. **Status:** the in-sample scoring is fixed by cross-fitting (section 3,
  requirement 4); the target framing is still debt (section 5).
- **M2. A new model goes live on every customer request.** Each job retrains, overwrites `model.pkl` and uses
  it immediately. There's no registry and no comparison with the previous model, so nobody can say which
  model produced a given set of predictions. This is the motivation for requirement 4.
- **M3. The headline metric is noisy and has no baseline.** It comes from one random 75/25 split with
  about 120 positives in the test set. That split gives AP 0.548, while 5-fold CV gives **0.520**. A
  no-model heuristic that **sorts pipes by age** already gets **AP 0.487**. Gradient boosting adds about 0.03
  AP over a rule a utility engineer could apply in a spreadsheet. Any promotion gate has to beat this
  heuristic, and not just the previous model, with an uncertainty estimate.
- **M4. `num_breaks_5yr` may be target leakage.** It is the second-strongest signal: removing it drops CV AP
  from 0.520 to 0.459. If its 5-year window overlaps the period in which `has_leak` was
  observed, a leak can also be counted as a break, and the feature then partly encodes the label. The data alone
  can't settle this, since 196 leaking rows in the raw file have zero breaks, so the relationship is not deterministic. It is an assumption
  to state and a question for the data owner (see A2).
- **M5. Preprocessing is fit on the whole dataset and refit on every batch.** Imputation medians, the network
  centroid (`dist_to_center_m`) and the density counts (`neighbours_500m`) are computed on whatever rows
  are in the file. At training time that leaks a little information from the test set. At scoring time it is
  **train/serve skew**: the same pipe gets different feature values depending on which other pipes are in the
  batch. This fitted state (medians, centroid, category list, CRS) belongs with the model artifact.
- **M6. Median imputation hides an informative gap.** Pipes with no `install_year` leak at **20%**
  against 15.8% for the rest. The sample is small (about 150 rows), but the signal points the right way,
  and median imputation removes it. A missing-value indicator should be added.
- **M7. The split ignores space.** Leaks and pipe characteristics can cluster spatially, so a random split
  can be optimistic. With spatial-block CV (1.5 km blocks) AP drops slightly, from 0.520 to 0.497. The effect
  on Fairport is small, but blocked or grouped splits should be the default once data from many utilities is pooled.
- **M8. The geo features add no measurable signal here.** Dropping `dist_to_center_m` and `neighbours_500m`
  gives AP 0.511 against 0.520, which is within noise. They are also the reason the image needs GDAL and geopandas,
  so they should earn their place or be dropped (see the decisions section).
- **M9. Impurity-based feature importances** (`feature_importances_`) are biased toward continuous,
  high-cardinality features. Permutation importance on held-out data is the honest version for analysts.
- **M10. One finding is fine as is.** Deduplication happens **before** the split, so duplicate rows do not leak across
  it. The 15 duplicates are exact copies. `keep="first"` would still silently pick a random row
  if two duplicates ever disagreed, so a check is worth adding.

### 1.6 Tests

- There are 3 tests, all on the cleaning functions, and all pass. Nothing covers the geo features, training,
  scoring, the stage contracts, or the failure paths in F1 and F2.
- `test_impute_missing_fills_with_material_median` only checks `notna()`, not that the value is the
  median.
- The tests rely on a `sys.path` hack instead of an installable package, and they aren't in the image.

---

### 1.7 Where each finding is addressed

| Findings | Fixed by (commit title) | How |
|---|---|---|
| F1, F2 | *Hotfix the live scripts* | `set -euo pipefail` (after `.bashrc`), and `elapsed` bound before `try`. Shipped first, so live customers benefit before any migration. |
| F3–F8, M2 | *Replace the scripts with a resumable stage runner…* | Per-job run directories, success markers with fingerprints, an append-only status log, real exceptions with exit codes. Customer jobs no longer train. |
| 1.4, M5, M6 | *Add the pipesense package core* | `as_of` date, per-utility UTM zone, unknown materials counted and capped, imputation fitted inside the model pipeline with missing-value indicators. Scoring does not need labels. |
| D1–D10, 1.2 | *Replace the shared conda env…*, *Rewrite the Dockerfile…* | uv lockfile with 13 runtime dependencies. Multi-stage, digest-pinned, non-root image with a `test` target. |
| M3, M4, M7, M9 | *Add the promotion gate…* plus the train stage | Spatial holdout, age-rule baseline, bootstrap CI, permutation importance. M4 is only partly fixed: see 5. Known debt. |
| M1 | *Score every pipe out-of-fold with cross-fitted models* | Every pipe is scored by a model that never trained on it, with `score_source` on every row, plus a `known_leak` flag. The target-framing half of M1 is debt: see 5. |
| `pickle` | train stage | MLflow saves the model with skops. Loading cannot execute code, and the trusted types are listed explicitly. |

---

## 2. Assumptions

The brief says not to ask, so each of these is a decision I would confirm with the team.

- **A1. A customer job means scoring, not training.** `{job_id, utility}` means "score this utility's current inventory
  with its serving model". Training is a separate pipeline that ends at the promotion gate. Retraining
  inside every customer request was the root of M2: unaudited models, slow jobs and non-reproducible results.
- **A2. `num_breaks_5yr` is observed before the leak window.** If the windows overlap, the feature
  leaks the label (M4). I kept it, because it is the second most important feature, but listing it in
  `[train].numeric_features` is a single config line. Removing it goes through the gate like any other
  model change, and that is my first question for the data owner.
- **A3. Every export carries an `as_of` date** (`raw/<utility>/meta.json`). Fairport's is set to
  2026-01-01, which reproduces the kit's `CURRENT_YEAR = 2026`, so ages did not change during the refactor.
- **A4. One registered model per utility** (`leak-risk-<utility>`). Utilities differ in materials,
  soil and data quality, and per-utility gating means one utility's bad data cannot regress another's
  model. A pooled model would help small utilities. The structure allows it later (see debt).
- **A5. Raw exports arrive in object storage** under `raw/<utility>/` and are never baked into images.
- **A6. Pipe IDs are unique within a utility**, and `has_leak` is 0/1 when present. Both are validated.
- **A7. The inspection budget is the riskiest 10% of pipes.** This is the business metric in the gate (`top_k_frac`).

---

## 3. Decisions (and the alternatives I weighed)

### Requirement 1: reproducible environment
- **uv with `uv.lock`**, instead of conda-lock, pip-tools or Poetry. Every dependency, including GDAL,
  PROJ and GEOS through the pyogrio, pyproj and shapely wheels, ships as a manylinux wheel, so conda adds nothing but solve
  time and the `defaults`-channel licensing question. uv gives one tool, hash-pinned transitive
  dependencies, `uv sync --frozen` in Docker and on laptops, and the dev/runtime split via dependency groups.
- **The team env's versions are kept** for numpy, pandas, scikit-learn and geopandas. Changing infrastructure and
  numerics in the same step would make any metric change undiagnosable. Upgrading them is a separate,
  gated change.
- **The runtime holds only what is imported**, plus `mlflow-skinny`, `skops`, `boto3` and `psycopg`. The full
  `mlflow` package (for a sqlite-backed registry in tests), `pytest` and `moto` are in the dev group only.

### Requirement 2: containers
- **One image, every stage** (`ENTRYPOINT ["pipesense"]`). The alternative was one image per stage. All stages share the same
  dependencies, so per-stage images would triple the build, scan and version surface for no isolation gain.
- **How stages run in sequence:** an in-process runner, with files as the hand-off. Each stage writes under
  `runs/<utility>/<job_id>/<stage>/` and commits with `_SUCCESS.json`. I considered Airflow (I run it in production),
  a broker between stages and Step Functions. For a two-stage job, an orchestrator adds a service to run and
  buys nothing the markers don't already give.
- **Resume after failure:** a stage is skipped when its marker's fingerprint matches. The fingerprint is
  a sha256 of the pipeline version, the stage's config section, the checksums of its raw inputs and the
  upstream fingerprints. Writing the marker last makes it the commit point: partial output is simply
  overwritten. A changed input re-runs exactly the affected stages. Tests: `test_resume_after_failure_skips_finished_stages`,
  `test_changed_raw_data_invalidates_prepare_and_downstream`, and the worker retry test.
- **Moving to another orchestrator without a rewrite:** stages are functions of a `StageContext` (utility, job_id,
  config, storage). They read no environment variables and use no globals or fixed paths. `pipesense stage <name> --job-id …` runs one
  stage after checking its upstream markers, and exit codes say whether to retry (0 / 2 permanent / 75 transient).
  An Airflow task, a Step Functions state or an AWS Batch job is a single command, and Batch's `evaluateOnExit`
  maps straight onto the exit codes.
- **Intermediate format:** Parquet instead of GeoPackage. Downstream stages need no geometry, and Parquet
  round-trips dtypes exactly.
- **Image versioning:** each image is tagged with its git SHA and carries OCI labels. `PIPESENSE_GIT_SHA` and `PIPESENSE_IMAGE_TAG` are written
  into every marker, model version and prediction manifest. Base images are pinned by digest. In AWS, ECR
  tags are immutable and deploys reference the digest.
- **A bad image in production:** roll back by redeploying the previous tag (in ECS, the previous task definition
  revision). Then find the affected outputs by the `image_tag` in their markers and manifests, and re-run those jobs
  with `--force`. Markers do **not** include the image tag on purpose: otherwise every deploy would invalidate
  every cached stage. When stage logic changes meaning, `[pipeline].version` is bumped instead.
- **Tests in the image:** `docker compose --profile test run --rm test` runs pytest in the `test` target,
  which is the runtime image plus dev dependencies and tests.

### Requirement 3: dev and production-like
- **Config, not code:** `config/dev.env` and `config/prod.env` select the artifact store (`file://` or `s3://`), the MLflow
  backend, the status backend and the log format. The AWS endpoints use boto3's standard `AWS_ENDPOINT_URL_S3/SQS`,
  which are simply absent on real AWS. There is no `if env == "prod"` anywhere in the code.
- **How a change travels:** commit → `test` image passes → build `pipesense:<sha>` → exercise it in dev
  → set the same `IMAGE_TAG` for prod-like. Prod services have no `build:` section and use `pull_policy: never`,
  so prod can only run an image that already exists. Pipeline parameters (`config/pipeline.toml`) are baked
  into the image, so gate thresholds travel with the code. **Models do not travel:** each environment trains on its own
  data and passes its own gate. The dev registry is disposable.
- **SeaweedFS instead of MinIO** for the local S3 API. MinIO stopped publishing its community images in 2026.

### Requirement 4: model lifecycle
- **Registry layout:** one MLflow registered model per utility. Aliases are used rather than the deprecated
  stages. Only the gate, or an audited manual `promote`, moves `champion`.
- **What winning means** (`gate.py`, thresholds in `[gate]`):
  - G0: scores are sane.
  - G1: the candidate beats ranking by age on average precision (AP) and on precision in the top 10%. A model that cannot beat a rule
    a utility engineer could apply in a spreadsheet should not ship.
  - G2: against the champion, AP improves by at least +0.005, without losing more than 0.02 precision@10% or 0.01 Brier score.
    The Brier check matters because customers read the scores as probabilities.
  - G3: the champion was evaluated on the same holdout definition, otherwise only a manual promote is possible.
  - A tie is rejected, so retraining unchanged data does not churn the champion. This is tested.
- **The holdout is the core of the design:** pipes are binned into 1 km blocks, and a hash of
  `(utility, block size, block)` puts about 20% of blocks into a holdout that **no version ever trains on**. Champion and
  challenger can therefore be compared fairly on the same unseen pipes, months apart, without storing
  holdout files. Spatial blocks rather than random rows, because neighbouring pipes share soil, age and
  pressure (M7). The gated model is trained on the non-holdout blocks only, and I accept losing about 20% of
  training data so that the audited numbers describe the deployed artifact.
- **Cross-fitted scoring (M1):** the registered model is the full model plus the 5 spatial CV fold models over
  the training blocks and the block→fold map, saved together with skops. A pipe in a training block is scored
  by the fold model that held that block out. Holdout pipes, and pipes in blocks that were empty at training
  time, are scored by the full model, which never trained on them. `score_source` (`oof` or `full`) is on
  every row. The gate still evaluates the full model on the holdout, unchanged. The fold models are the ones
  that produce the spatial CV AP (0.523), so every audited number now describes scores a customer actually
  receives. Training cost is unchanged: the fold models were already fitted for the CV metric and then thrown
  away. I rejected the alternatives of keeping in-sample scores with a flag (the list stays misleading) and
  leaving training pipes out of the output (a utility needs a score for every pipe). The cost is that one
  ranking mixes scores from 6 similar models. Because the full model is unchanged, the gate sees a
  cross-fitted candidate as a tie with an older single-model champion, so the switch is an audited manual
  `promote`.
- **The bootstrap CI is reported, not gated on.** With about 160 holdout leaks the 90% CI of the AP lift is roughly
  ±0.05. Requiring the lower bound to be above 0 would block almost every honest improvement on one utility's
  data. The CI is in every gate report, so a reviewer can see when a promotion was marginal.
- **Audit trail:** each decision is an MLflow run (`leak-risk/<utility>/gate`). It holds every check with value and
  threshold, all metrics, the bootstrap, the rules and the actor (`gate:auto` or a person). The run ID is tagged on
  the model version, together with lineage tags (data fingerprint, git SHA, job). MLflow keeps no alias history, so
  these runs are the history.
- **Every prediction batch records its model:** `model_name`, `model_version` and `score_source` are on every
  row of `predictions.csv`, and `manifest.json` holds the run ID, data fingerprint, `as_of`, git SHA, image
  tag and the count of `oof` and `full` scores. The champion alias is resolved once, then the job loads that
  exact version, so the recorded version is the one that scored even if the alias moves mid-job.
- **Drift monitoring:** at train time the model's run stores a profile of its training rows as the model sees
  them: decile bins plus a missing share for numeric features, and category shares for categoricals. Every
  scoring job computes PSI per feature against the **champion's own** profile, writes it to `manifest.json`
  and logs a warning above 0.2 (`[predict].psi_warn_threshold`). It is a warning, not a failure: drift is a
  reason to look and to retrain, not to withhold a customer's results. Bad data is already refused by
  `prepare`. PSI was chosen over KS tests because it treats numeric and categorical features the same way, and
  because at 3,000 rows a KS test flags differences too small to matter. On Fairport, scoring the full
  inventory gives PSI ≤ 0.01 for every feature. The unseen holdout blocks on their own reach about 0.1 on the
  two geo features, and an inventory five years older gives 0.05 on age, so 0.2 means a real change to the
  network or the export. Champions trained before this change have no profile, and their jobs record
  `"status": "no_reference"`.
- **Serialisation:** skops (MLflow's default) instead of pickle, with the trusted types listed explicitly.
  `Tree` is trusted because models only come from our own registry.

### Bonus: the miniature
- **Queue:** ElasticMQ, an SQS-compatible broker. RabbitMQ would have worked, but with ElasticMQ the worker uses the
  same boto3 calls, visibility timeouts, receive counts and DLQ as the SQS in DESIGN.md. The miniature is
  the design, not an analogy.
- **Failure policy:**
  - Permanent errors fail fast to the DLQ with the reason. Retrying bad data wastes compute and delays telling the customer.
  - Transient errors retry with exponential visibility backoff and resume from markers, then dead-letter after 3 attempts.
  - A crashed worker's message is redelivered automatically.
  - A status-store outage never fails a job.
  - The demo shows a transient fault (attempt 2 skips `prepare`) and a broken export (`brokenville`, straight to the DLQ).
- **Status:** an append-only Postgres event log, served by a ~100-line stdlib page that refreshes itself. Grafana or a
  web framework would be one more thing to run, with no new information.
- **Fault injection** is a message field honoured only when `PIPESENSE_ALLOW_FAULT_INJECTION=true` (local only).

### Requirement 5
See DESIGN.md. In short: an SQS fair queue → autoscaled ECS Fargate workers (the same worker), RDS, S3 and MLflow.
Step Functions + Batch is the documented alternative. Migration is per-utility routing after shadow runs.

---

## 4. ML results on Fairport (production config)

- **Holdout:** 945 pipes, 159 leaking (1 km blocks, 20% of blocks; the realised share varies with block density).
- **Candidate v1:** AP 0.495, ROC AUC 0.83, precision@10% 0.574, Brier 0.112. Spatial 5-fold CV AP on the
  training blocks is 0.523.
- **Age rule:** AP 0.429, precision@10% 0.511. Lift +0.064 AP, bootstrap 90% CI [+0.019, +0.112]. **PROMOTED.**
- **Retraining the same data:** v2 has AP lift +0.000 over the champion. **REJECTED** (no churn).
- **Out-of-fold scores** (dev run, v1 single model → v2 cross-fitted, same data): measured against
  `known_leak`, the delivered list's AP fell from 0.785 (in-sample) to 0.512, in line with the CV (0.523) and
  holdout (0.495) AP. The list now looks as good as it really is. The top 25 went from 25 known leaks to 19.
  12 pipes stayed and 13 entered, and the 6 entrants without a recorded leak are cast iron aged 72 to 123
  years. Known leaks fell from 93 to 71 in the top 100, and from 258 to 171 in the top 10%. Holdout pipes keep
  identical scores, and they now take 82 rather than 72 of the top-10% places, because training pipes no
  longer get inflated in-sample scores. Rank correlation between the two lists is 0.93.
- **Permutation importance** (holdout AP drop): age 0.22, `num_breaks_5yr` 0.08, `dist_to_center_m` 0.05,
  material 0.05, then soil, length, pressure zone, density. Diameter is about 0. This settles M8: distance to centre
  earns its place on the holdout, and density does not, but dropping it is a gated model change, not a refactor.
- **Honest summary:** the model is a modest but real improvement over the obvious rule. The next gains come
  from data (leak dates, pressure transients, repair history), not from model tuning.

---

## 5. Known debt (what I would do next, in order)

1. **Target framing (M1).** In-sample scoring is fixed: no pipe is scored by a model that trained on it. What
   remains is that `has_leak` is a historical label, so the model ranks pipes that resemble pipes that have
   leaked, not pipes about to leak. The real fix needs leak *dates*: train on features as of T with the label
   "leak in (T, T+12 months]", and use a temporal plus spatial holdout. Until then `known_leak` lets a utility
   filter the list. Cross-fitting also routes by block, not by pipe: if a pipe's coordinates are corrected
   after training and it moves into another training block, it is scored by a fold model that saw it. Routing
   known training pipes by `pipe_id` first would close that gap.
2. **`num_breaks_5yr` (M4)** needs the data owner's window definition before the next promotion.
3. **Network-level features** (centroid, density) depend on the inventory at scoring time. They are consistent
   because jobs score the full inventory, but they drift as the network grows. They should become a versioned
   per-utility network snapshot.
4. **Train idempotency:** if `train` dies after registering a version but before its marker, the retry
   registers another version. The orphan is harmless (it is never gated or served), but it should be found by
   `job_id` and reused.
5. **Two workers on one job** (a heartbeat failure followed by redelivery) would write the same deterministic outputs. In
   AWS I would add a conditional-write lease per `job_id` (a Postgres row lock or a DynamoDB condition).
6. **Image size:** about 700 MB of venv, dominated by pyarrow, scipy and the bundled GDAL in pyogrio. It is fine for batch. A distroless
   base and dropping matplotlib (log importances as a table only) would roughly halve it.
7. **Small-utility models:** a pooled multi-utility model with utility-level features, gated per utility.
8. **No CI pipeline** (out of scope per the brief). `docker compose --profile test run --rm test` is the CI step.
9. **The MLflow server image** pins pip versions but not hashes. The compose credentials are local-only placeholders.

---

## 6. AI usage

I used Claude (Anthropic) as a pair-programmer throughout:
- **Review:** it ran the untouched pipeline and reproduction scripts with me. I checked every finding against the code
  and kept only what reproduced (for example, the stale-features SUCCESS run and the `elapsed` NameError).
- **Implementation:** drafted with Claude in small commits.
  I reviewed every diff, ran the tests and the end-to-end flows, and made the design calls (score vs train
  split, the gate rules, spatial holdout, failure policy, SQS + ECS over Batch).
- **Verification without a Docker daemon** (in Claude's sandbox): the builder step was reproduced with
  `uv sync --frozen --no-dev --no-editable`. The prod-like profile was run natively against Postgres 16, a moto
  S3+SQS server and an MLflow server on Postgres. I then ran the compose profiles on my machine.
- Where AI output was wrong or risky, I changed it. For example, strict mode broke on `~/.bashrc`, and MLflow 3.16
  rejected skops' untrusted types until the reviewed list was added.
