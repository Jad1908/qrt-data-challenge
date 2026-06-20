"""Single source of truth: paths, schema constants, and the experiment spec.

Nothing here does I/O or heavy work — it only declares facts and configuration.
Every other module imports its column names and constants from this file so that
the schema is defined exactly once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# ── Paths ──────────────────────────────────────────────────────────────────
# parents[1] = the project root (this file lives in src/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# The competition ships hashed filenames (e.g. X_train_9xQjqvZ.csv).
# We store the *patterns* here; the actual resolution-by-glob happens in data.py,
# which is the only module allowed to know about raw-file quirks.
X_TRAIN_GLOB = "X_train*.csv"
Y_TRAIN_GLOB = "y_train*.csv"
X_TEST_GLOB = "X_test*.csv"
SAMPLE_SUBMISSION_GLOB = "sample_submission*.csv"

# ── Schema: raw columns ────────────────────────────────────────────────────
ID_COL = "ROW_ID"
TS_COL = "TS"
ALLOCATION_COL = "ALLOCATION"
GROUP_COL = "GROUP"
TURNOVER_COL = "MEDIAN_DAILY_TURNOVER"
TARGET_COL = "target"  # lives in y_train; lower-cased on load

N_LAGS = 20
RET_COLS = [f"RET_{i}" for i in range(1, N_LAGS + 1)]  # RET_1 = most recent
VOL_COLS = [f"SIGNED_VOLUME_{i}" for i in range(1, N_LAGS + 1)]  # likewise

# ── Schema: columns derived by data.py ─────────────────────────────────────
TS_ORDER_COL = "ts_order"  # int parsed from DATE_0001 -> 1  (chronological)
ALLOC_ORDER_COL = "alloc_order"  # int parsed from ALLOCATION_07 -> 7
TARGET_SIGN_COL = "target_sign"  # 1 if target > 0 else 0  (the accuracy label)

# ── Experiment specification ───────────────────────────────────────────────
Task = Literal["classify", "regress"]


@dataclass
class CVConfig:
    """Knobs for the date-aware cross-validation (built in commit 3).

    Note there is no shuffle seed: a forward time-split is deterministic by
    construction. `embargo` is measured in *dates* and should be >= the lookback
    window so a validation fold's 20-day history never overlaps training rows.
    """

    n_splits: int = 5
    embargo: int = N_LAGS
    scheme: Literal["expanding", "rolling"] = "expanding"


@dataclass
class ExperimentConfig:
    """A complete, printable description of one run.

    The whole point: an experiment is a *value* you can log, diff, and reproduce
    — not a pile of mutable notebook state. You explore by editing this object.
    """

    name: str = "baseline"
    task: Task = "classify"  # the classify/regress fork
    features: list[str] = field(default_factory=lambda: ["ret_summary"])
    model: str = "logreg"  # resolved by models.py registry
    model_params: dict[str, Any] = field(default_factory=dict)
    cv: CVConfig = field(default_factory=CVConfig)
    seed: int = 42

    def __post_init__(self) -> None:
        if self.task not in ("classify", "regress"):
            raise ValueError(f"task must be 'classify' or 'regress', got {self.task!r}")

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for logging / experiment tracking."""
        return asdict(self)
