"""Stage registry and pipeline definitions.

* score: a customer job. It scores the utility's current inventory with the serving (champion) model.
* train: produces a candidate model and puts it through the promotion gate. It never runs as
  part of a customer job.
"""

from pipesense.runner import Stage
from pipesense.stages import gate, predict, prepare, train

STAGES: dict[str, Stage] = {
    "prepare": Stage("prepare", prepare.run, config_sections=("prepare",), external_inputs=prepare.input_checksums),
    "train": Stage("train", train.run, upstream=("prepare",), config_sections=("train",)),
    "gate": Stage("gate", gate.run, upstream=("prepare", "train"), config_sections=("gate", "train")),
    "predict": Stage("predict", predict.run, upstream=("prepare",), config_sections=("predict",)),
}

PIPELINES: dict[str, list[str]] = {
    "score": ["prepare", "predict"],
    "train": ["prepare", "train", "gate"],
}
