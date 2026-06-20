"""The data contract.

This is the ONLY module that touches raw files or knows about their quirks
(hashed filenames, column casing, the X/y split). Everything downstream consumes
the canonical frame returned here and never re-reads a CSV.

Canonical frame guarantees (relied on by every other module):
  * sorted by (ts_order, alloc_order), index reset to 0..n-1
  * ts_order / alloc_order are ints parsed from the anonymized labels
  * floats downcast to float32
  * train frame carries `target` and `target_sign`; test frame carries neither
  * ROW_ID preserved on both so predictions can be mapped back for submission
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config as C


# ── File resolution ─────────────────────────────────────────────────────────
def _resolve(glob_pattern: str) -> Path:
    """Resolve a hashed filename to exactly one path, or fail clearly."""
    matches = sorted(C.DATA_DIR.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {glob_pattern!r} in {C.DATA_DIR}")
    if len(matches) > 1:
        raise ValueError(
            f"Expected one file for {glob_pattern!r}, found {len(matches)}: "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV and downcast float64 -> float32 to halve memory."""
    df = pd.read_csv(path)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


# ── Shared canonicalization ──────────────────────────────────────────────────
def _add_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse anonymized TS / ALLOCATION labels into orderable integers, sort.

    'DATE_0001' -> 1, 'ALLOCATION_07' -> 7. We extract first and check for
    unparseable values *before* casting, so a malformed label yields a clear
    error instead of an opaque pandas cast failure.
    """
    ts_num = df[C.TS_COL].astype(str).str.extract(r"(\d+)", expand=False)
    alloc_num = df[C.ALLOCATION_COL].astype(str).str.extract(r"(\d+)", expand=False)

    if ts_num.isna().any():
        bad = df.loc[ts_num.isna(), C.TS_COL].unique()[:5]
        raise ValueError(f"Unparseable {C.TS_COL} values, e.g. {bad}")
    if alloc_num.isna().any():
        bad = df.loc[alloc_num.isna(), C.ALLOCATION_COL].unique()[:5]
        raise ValueError(f"Unparseable {C.ALLOCATION_COL} values, e.g. {bad}")

    df[C.TS_ORDER_COL] = ts_num.astype("int32")
    df[C.ALLOC_ORDER_COL] = alloc_num.astype("int32")

    return df.sort_values([C.TS_ORDER_COL, C.ALLOC_ORDER_COL]).reset_index(drop=True)


# ── Caching ──────────────────────────────────────────────────────────────────
def _cached(builder, cache_name: str, use_cache: bool) -> pd.DataFrame:
    """Parquet-cache wrapper. Delete the cache dir to invalidate.

    Caveat: the cache is keyed only by name, NOT by source-file contents. If the
    raw CSVs change, delete cache/ manually. Parquet preserves our dtypes
    (float32, int32) so a cached load is identical to a fresh build.
    """
    path = C.CACHE_DIR / cache_name
    if use_cache and path.exists():
        return pd.read_parquet(path)
    df = builder()
    if use_cache:
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df


# ── Public API ───────────────────────────────────────────────────────────────
def load_train(use_cache: bool = True) -> pd.DataFrame:
    """Canonical training frame: X merged with y, ordered, contract-checked."""

    def build() -> pd.DataFrame:
        X = _read_csv(_resolve(C.X_TRAIN_GLOB))
        y = _read_csv(_resolve(C.Y_TRAIN_GLOB))

        # Normalize y to exactly [ROW_ID, target]. The y file has two columns;
        # one is the id, the other is the label (sample ships it lower-cased).
        y = y.rename(columns={c: c.strip() for c in y.columns})
        non_id = [c for c in y.columns if c != C.ID_COL]
        if len(non_id) != 1:
            raise ValueError(f"Expected one label column in y, got {non_id}")
        y = y.rename(columns={non_id[0]: C.TARGET_COL})[[C.ID_COL, C.TARGET_COL]]

        # one_to_one validates ROW_ID is unique on BOTH sides and the merge is 1:1.
        df = X.merge(y, on=C.ID_COL, how="left", validate="one_to_one")
        if df[C.TARGET_COL].isna().any():
            n = int(df[C.TARGET_COL].isna().sum())
            raise ValueError(f"{n} train rows have no matching target")

        df = _add_order_columns(df)
        # The accuracy label: sign(x)=1 iff x>0, else 0 (ties -> 0 / bet against).
        df[C.TARGET_SIGN_COL] = (df[C.TARGET_COL] > 0).astype("int8")
        return df

    return _cached(build, "train.parquet", use_cache)


def load_test(use_cache: bool = True) -> pd.DataFrame:
    """Canonical test frame: same shape as train minus target columns."""

    def build() -> pd.DataFrame:
        X = _read_csv(_resolve(C.X_TEST_GLOB))
        return _add_order_columns(X)

    return _cached(build, "test.parquet", use_cache)
