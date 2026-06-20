"""Submission formatting — isolated because format bugs are dumb and frequent.

Validates against the sample template (column names, ROW_ID coverage, label
domain) before writing, so a malformed file fails here, loudly, not on upload.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data import _resolve  # reuse the one file-resolution helper


def _sample_template() -> pd.DataFrame:
    return pd.read_csv(_resolve(C.SAMPLE_SUBMISSION_GLOB))


def build_submission(row_id: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Map predictions onto the exact schema of the sample submission."""
    template = _sample_template()
    pred_col = [c for c in template.columns if c != C.ID_COL]
    if len(pred_col) != 1:
        raise ValueError(f"Unexpected submission columns: {list(template.columns)}")
    pred_col = pred_col[0]

    sub = pd.DataFrame({C.ID_COL: row_id, pred_col: np.asarray(pred).astype(int)})

    # Validate coverage and ordering against the template.
    want = set(template[C.ID_COL])
    got = set(sub[C.ID_COL])
    if want != got:
        raise ValueError(
            f"ROW_ID mismatch: missing {len(want - got)}, extra {len(got - want)}"
        )
    if not set(sub[pred_col].unique()) <= {0, 1}:
        raise ValueError("Predictions must be 0/1.")

    # Reorder to the template's exact row order.
    order = {rid: i for i, rid in enumerate(template[C.ID_COL])}
    sub = sub.sort_values(C.ID_COL, key=lambda s: s.map(order)).reset_index(drop=True)
    return sub[[C.ID_COL, pred_col]]


def write_submission(row_id: np.ndarray, pred: np.ndarray, name: str) -> Path:
    sub = build_submission(row_id, pred)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = C.OUTPUT_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    return path
