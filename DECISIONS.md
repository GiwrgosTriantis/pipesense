# DECISIONS

This file has five parts: findings from the code review, assumptions, decisions, known debt and AI usage.
I did the review on the untouched starter kit, before changing anything. Every finding with a number
was reproduced by running the original code (see `analysis/review_checks.py`).

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
  model did not see in training.
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

## 2. Assumptions

These are proposed and still need confirming. Each one changes a design choice.

- **A1. What a job is.** A customer request for `{job_id, utility}` means "score this utility's current pipe inventory
  with the serving model". Training is a separate, scheduled or manual flow that ends at the promotion
  gate, not something that happens inside every customer job (fixes M2).
- **A2. `has_leak` and `num_breaks_5yr`.** I assume `has_leak` means "a leak was recorded in the observation window" and
  `num_breaks_5yr` counts breaks *before* that window. If they overlap, `num_breaks_5yr` must be dropped or
  recomputed with a cutoff (M4). I will train and report both variants.
- **A3. Snapshot date.** Each raw export carries, or is given, an `as_of` date, and age is computed from it.
- **A4. Global model or one per utility.** _TBD: decide when building requirement 4._
- **A5. Data location.** Raw data arrives through object storage (S3, or MinIO locally), never baked into the image.

## 3. Decisions (alternatives weighed)

_Fill in while building. One entry per decision: what I chose, the alternatives, why, and what would change my mind._

## 4. Known debt

_Fill in at the end._

## 5. AI usage

_Fill in honestly at the end. For example, Claude helped with the code review and ran the reproduction checks
(`analysis/review_checks.py`), and I verified every finding against the code myself._
