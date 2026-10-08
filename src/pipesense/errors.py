"""Error taxonomy.

Each error class maps to a process exit code. An orchestrator (the local worker, AWS Batch or
Airflow) can then decide whether to retry from the exit code alone, without parsing logs.
"""

EXIT_OK = 0
EXIT_PERMANENT = 2  # bad input or missing prerequisite: retrying the same job cannot help
EXIT_TRANSIENT = 75  # EX_TEMPFAIL: infrastructure hiccup, safe to retry


class PipelineError(Exception):
    """Base class. Unclassified errors are treated as transient (retry, then dead-letter)."""

    exit_code = EXIT_TRANSIENT
    retryable = True


class PermanentError(PipelineError):
    """Retrying the same job with the same inputs will fail the same way."""

    exit_code = EXIT_PERMANENT
    retryable = False


class DataValidationError(PermanentError):
    """The utility's input data violates the data contract."""


class MissingPrerequisiteError(PermanentError):
    """Something the stage needs does not exist (no champion model, upstream stage not run)."""


class TransientError(PipelineError):
    """Explicitly retryable failure (timeouts, throttling, injected faults in dev)."""
