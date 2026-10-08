# PipeSense — leak pipeline (starter kit)

This is the batch pipeline: every time a customer requests an analysis, the job watcher launches one run of it. This copy is set up for the Fairport water utility network.

## What it does

```
data/pipes.csv ──▶ prepare_data.py ──▶ output/features.gpkg
                                          │
                          train_model.py ─┴─▶ output/model.pkl, feature_importances.png
                                          │
                              predict.py ─┴─▶ output/predictions.csv, status.json
```

- **`pipeline/prepare_data.py`** — cleans the raw GIS export (duplicate ids, mixed
  material spellings, missing years, bad lengths), builds geospatial features
  (distance to network center, pipe density within 500 m) and writes a GeoPackage.
- **`pipeline/train_model.py`** — trains a gradient boosting classifier on the
  feature table, prints ROC AUC / average precision, pickles the model.
- **`pipeline/predict.py`** — scores all pipes and writes `predictions.csv`
  sorted by risk.
- **`pipeline/common.py`** — shared helpers (timing, status file, csv I/O).
- **`run_all.sh`** — what the job watcher runs (on the VM) for each requested job.
- **`environment.yml`** — the team's conda environment (everyone shares this one).
- **`Dockerfile`** — a first containerization attempt; it "works" but nobody is happy with it.
- **`tests/`** — the tests we have. Run with `pytest tests/ -v`.

## How the team runs it today

```bash
conda env create -f environment.yml
conda activate pipesense
bash run_all.sh
```

Stages communicate through files in `WORK_DIR` (default `./output`). On the VM,
`WORK_DIR` points at a shared NFS mount that a monitor process polls for
`status.json`.

More details can be found in `ASSESSMENT.md`.
