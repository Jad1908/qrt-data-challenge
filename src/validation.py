"""Purged, embargoed, date-grouped forward cross-validation.

This is the only object that defines what "train" and "val" mean. Every fold is
a forward split (train = past dates, val = future dates) with an embargo gap that
removes the most recent training dates near the boundary, so a validation row's
20-day lookback window can never overlap the training period or a training label.

Units: `embargo` is counted in UNIQUE DATES (positions in the sorted date list),
not in ts_order integer values — the lookback is 20 trading days = 20 positions,
which is correct even if the anonymized date numbering has gaps.

See module commit notes for the two leakage channels; `embargo = N_LAGS` is the
exact minimum that closes both.

Yielded indices are POSITIONAL (0..n-1 into the canonical frame) — use with
DataFrame.iloc / numpy indexing. This relies on data.py's reset_index.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config as C


@dataclass
class FoldInfo:
    """Human-readable description of one fold, for the describe() diagnostic."""

    fold: int
    n_train_dates: int
    n_val_dates: int
    n_train_rows: int
    n_val_rows: int
    train_date_first: int
    train_date_last: int
    val_date_first: int
    val_date_last: int
    embargo_gap_dates: int


class PurgedTimeSeriesSplit:
    """Forward CV over dates with a leakage embargo.

    Parameters
    ----------
    n_splits : number of validation folds.
    embargo  : unique dates dropped between train and val. Should be >= N_LAGS;
               a warning fires if it is smaller, since the lookback would leak.
    scheme   : 'expanding' (train = all past minus embargo) or
               'rolling' (train = a fixed window just before the embargo gap).

    The timeline of T unique dates is cut into (n_splits + 1) contiguous blocks.
    The first block is the initial training seed and is never validated; blocks
    1..n_splits are the validation folds (the last absorbs any remainder).
    """

    def __init__(
        self, n_splits: int = 5, embargo: int = C.N_LAGS, scheme: str = "expanding"
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if scheme not in ("expanding", "rolling"):
            raise ValueError(f"scheme must be 'expanding' or 'rolling', got {scheme!r}")
        if embargo < C.N_LAGS:
            warnings.warn(
                f"embargo={embargo} < N_LAGS={C.N_LAGS}: validation lookback "
                f"windows will overlap training — results may be leak-inflated.",
                stacklevel=2,
            )
        self.n_splits = n_splits
        self.embargo = embargo
        self.scheme = scheme

    def get_n_splits(self) -> int:
        return self.n_splits

    @classmethod
    def from_config(cls, cv: "C.CVConfig") -> "PurgedTimeSeriesSplit":
        return cls(n_splits=cv.n_splits, embargo=cv.embargo, scheme=cv.scheme)

    # ── internals ────────────────────────────────────────────────────────────
    def _date_layout(self, df: pd.DataFrame):
        """Return (sorted unique dates, {date_value: positional row indices})."""
        unique_dates = np.sort(df[C.TS_ORDER_COL].unique())
        date_to_rows = df.groupby(C.TS_ORDER_COL).indices  # value -> ndarray of row pos
        return unique_dates, date_to_rows

    def _rows_for(self, unique_dates, date_to_rows, lo: int, hi: int) -> np.ndarray:
        """Gather sorted row positions for date-positions [lo, hi)."""
        if hi <= lo:
            return np.empty(0, dtype=int)
        parts = [date_to_rows[unique_dates[p]] for p in range(lo, hi)]
        return np.sort(np.concatenate(parts))

    def _bounds(self, T: int):
        """Yield (val_start_pos, val_end_pos) for each fold, in date-positions."""
        fold_size = T // (self.n_splits + 1)
        if fold_size < 1:
            raise ValueError(
                f"Too few unique dates ({T}) for n_splits={self.n_splits}."
            )
        for i in range(self.n_splits):
            val_start = (i + 1) * fold_size
            val_end = (i + 2) * fold_size if i < self.n_splits - 1 else T
            yield val_start, val_end

    # ── public API ───────────────────────────────────────────────────────────
    def split(self, df: pd.DataFrame):
        """Yield (train_idx, val_idx) positional arrays, fold by fold."""
        unique_dates, date_to_rows = self._date_layout(df)
        T = len(unique_dates)

        for fold, (val_start, val_end) in enumerate(self._bounds(T)):
            train_end = val_start - self.embargo  # exclusive; the embargo gap
            if self.scheme == "expanding":
                train_start = 0
            else:  # rolling: fixed window of one block-width before the gap
                window = T // (self.n_splits + 1)
                train_start = max(0, train_end - window)

            if train_end - train_start < 1:
                raise ValueError(
                    f"Fold {fold}: empty training set (embargo={self.embargo} too "
                    f"large for the fold size). Reduce n_splits or embargo."
                )

            train_idx = self._rows_for(
                unique_dates, date_to_rows, train_start, train_end
            )
            val_idx = self._rows_for(unique_dates, date_to_rows, val_start, val_end)
            yield train_idx, val_idx

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tabulate every fold for a one-glance sanity check (no model involved)."""
        unique_dates, date_to_rows = self._date_layout(df)
        T = len(unique_dates)
        rows: list[FoldInfo] = []

        for fold, (val_start, val_end) in enumerate(self._bounds(T)):
            train_end = val_start - self.embargo
            train_start = (
                0
                if self.scheme == "expanding"
                else max(0, train_end - T // (self.n_splits + 1))
            )

            tr = self._rows_for(unique_dates, date_to_rows, train_start, train_end)
            va = self._rows_for(unique_dates, date_to_rows, val_start, val_end)

            rows.append(
                FoldInfo(
                    fold=fold,
                    n_train_dates=train_end - train_start,
                    n_val_dates=val_end - val_start,
                    n_train_rows=len(tr),
                    n_val_rows=len(va),
                    train_date_first=int(unique_dates[train_start]),
                    train_date_last=int(unique_dates[train_end - 1]),
                    val_date_first=int(unique_dates[val_start]),
                    val_date_last=int(unique_dates[val_end - 1]),
                    embargo_gap_dates=val_start - train_end,
                )
            )
        return pd.DataFrame([r.__dict__ for r in rows])
