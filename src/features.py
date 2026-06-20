"""Feature registry + tiered computation machinery.

A feature's TIER determines its interface, and its interface determines WHERE the
orchestrator may compute it — this is the structural enforcement of leak-safety:

  Tier 0  row-local            stateless  computed once on the full frame
  Tier 1  same-date x-section  stateless  computed once on the full frame
  Tier 2  across-time/target   STATEFUL  fit() on a fold's train rows, only inside CV

Stateless features implement transform(df) -> DataFrame(new cols, aligned index).
Stateful features additionally implement fit(train_df); their transform() reads
ONLY non-target columns (TS_ORDER, ALLOC_ORDER, ...), never the label — so there
is no path by which a validation target can reach the model.

Each feature is INDEPENDENT: it recomputes whatever it needs from raw columns and
never depends on another feature's output, so any subset in cfg.features works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

EPS = 1e-9


# ── Base + registry ──────────────────────────────────────────────────────────
class Feature:
    """Base class. `tier` drives statefulness; `outputs` lists produced columns."""

    name: str = "base"
    tier: int = 0
    outputs: list[str] = []

    @property
    def stateful(self) -> bool:
        return self.tier >= 2

    def fit(self, df: pd.DataFrame) -> "Feature":
        """No-op for stateless features; overridden by Tier-2."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


_REGISTRY: dict[str, type[Feature]] = {}


def register(cls: type[Feature]) -> type[Feature]:
    if cls.name in _REGISTRY:
        raise ValueError(f"Duplicate feature name {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def make_feature(name: str, **params) -> Feature:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown feature {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def available_features() -> dict[str, int]:
    """name -> tier, for discovery / sanity printing."""
    return {n: cls.tier for n, cls in sorted(_REGISTRY.items())}


# ── Tier 0: row-local return summary ─────────────────────────────────────────
@register
class RetSummary(Feature):
    """Momentum / reversal / volatility descriptors from the RET block.

    All row-local: a row's outputs use only its own RET_1..20. RET_1 is the most
    recent day (yesterday's realized return), RET_20 the oldest.
    """

    name = "ret_summary"
    tier = 0
    outputs = ["ret_mean_5", "ret_mean_20", "ret_std_20", "ret_last", "ret_sharpe_20"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = df[C.RET_COLS]
        recent5 = [f"RET_{i}" for i in range(1, 6)]
        mean20 = ret.mean(axis=1)
        std20 = ret.std(axis=1)
        out = pd.DataFrame(index=df.index)
        out["ret_mean_5"] = df[recent5].mean(axis=1)  # short-horizon (reversal-prone)
        out["ret_mean_20"] = mean20  # long-horizon (momentum)
        out["ret_std_20"] = std20  # realized vol
        out["ret_last"] = df["RET_1"]  # yesterday alone
        out["ret_sharpe_20"] = mean20 / (std20 + EPS)  # risk-adjusted drift
        return out.astype("float32")


# ── Tier 1: same-date cross-sectional ────────────────────────────────────────
@register
class RetCrossSection(Feature):
    """This allocation's recent performance RELATIVE to its peers that same day.

    Stateless and leak-safe: every output for a row at date t uses only other
    rows at date t (and no targets). Computing on the full frame equals computing
    per-fold, because a date never straddles the train/val boundary.
    """

    name = "ret_cross_section"
    tier = 1
    outputs = ["ret_mkt", "ret_demeaned", "ret_rank"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df[C.RET_COLS].mean(axis=1)  # recompute, no cross-feature dep
        g = base.groupby(df[C.TS_ORDER_COL])
        out = pd.DataFrame(index=df.index)
        out["ret_mkt"] = g.transform("mean")  # "effet marché" that day
        out["ret_demeaned"] = base - out["ret_mkt"]  # relative strength (signed)
        out["ret_rank"] = g.rank(pct=True)  # relative strength (rank, 0..1)
        return out.astype("float32")


# ── Tier 2: expanding per-allocation hit-rate (the dangerous-feature template) ─
@register
class AllocHitRate(Feature):
    """P(target>0) for this allocation, estimated from PAST training rows only.

    This is time-aware target encoding. For a row at (allocation S, date t):
        estimate = (sum of target_sign over TRAINING rows of S with date < t
                    + m * prior) / (count of those rows + m)
    with `prior` = global training base rate and `m` = smoothing pseudo-count
    (shrinks low-history allocations toward the prior to avoid overfitting).

    Why it's leak-free:
      * fit() is only ever called on a fold's TRAINING rows (orchestrator's job).
      * transform() reads only TS_ORDER and ALLOC_ORDER — never a target.
      * searchsorted(..., 'left') counts rows with date STRICTLY < t, so a
        training row never sees its own label, and (forward split) a validation
        row at date v sees exactly the full training history for S, nothing later.
    """

    name = "alloc_hit_rate"
    tier = 2
    outputs = ["alloc_hit_rate"]

    def __init__(self, smoothing: float = 50.0):
        self.m = float(smoothing)
        self.prior_: float = 0.5
        self.store_: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, df: pd.DataFrame) -> "AllocHitRate":
        self.prior_ = float(df[C.TARGET_SIGN_COL].mean())
        self.store_ = {}
        sign = df[C.TARGET_SIGN_COL].to_numpy()
        ts = df[C.TS_ORDER_COL].to_numpy()
        for alloc, pos in df.groupby(C.ALLOC_ORDER_COL).indices.items():
            d, s = ts[pos], sign[pos]
            order = np.argsort(d, kind="stable")  # ensure date-ascending
            d = d[order]
            # prefix[k] = sum of the first k target_signs in date order
            prefix = np.concatenate([[0.0], np.cumsum(s[order])]).astype("float64")
            self.store_[alloc] = (d, prefix)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = np.full(len(df), self.prior_, dtype="float64")
        ts = df[C.TS_ORDER_COL].to_numpy()
        for alloc, pos in df.groupby(C.ALLOC_ORDER_COL).indices.items():
            store = self.store_.get(alloc)
            if store is None:  # unseen allocation -> prior
                continue
            dates_sorted, prefix = store
            k = np.searchsorted(dates_sorted, ts[pos], side="left")  # # past rows
            s = prefix[k]  # sum past signs
            result[pos] = (s + self.m * self.prior_) / (k + self.m)
        return pd.DataFrame({self.name: result}, index=df.index).astype("float32")


# ── Pipeline ─────────────────────────────────────────────────────────────────
class FeaturePipeline:
    """Assembles chosen features and separates the eager path from the per-fold one.

    Orchestrator contract (commit 7 will use exactly this):
        pipe = FeaturePipeline(cfg.features)
        stateless = pipe.transform_stateless(full_frame)   # ONCE, safe
        ... fold loop ...
            pipe.fit_stateful(train_rows)                   # inside the fold only
            tr = pipe.transform_stateful(train_rows)
            va = pipe.transform_stateful(val_rows)
        X = stateless[cols] (+) stateful columns,  selected via pipe.output_columns
    """

    def __init__(self, feature_names: list[str], params: dict | None = None):
        params = params or {}
        self.features = [make_feature(n, **params.get(n, {})) for n in feature_names]
        self.stateless = [f for f in self.features if not f.stateful]
        self.stateful = [f for f in self.features if f.stateful]

    @property
    def output_columns(self) -> list[str]:
        cols: list[str] = []
        for f in self.features:
            cols += f.outputs
        return cols

    def _concat(self, df: pd.DataFrame, feats: list[Feature]) -> pd.DataFrame:
        if not feats:
            return pd.DataFrame(index=df.index)
        return pd.concat([f.transform(df) for f in feats], axis=1)

    def transform_stateless(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 0/1 columns. Safe to call once on the full (train+test) frame."""
        return self._concat(df, self.stateless)

    def fit_stateful(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """Fit every Tier-2 feature on this fold's training rows. CV-loop only."""
        for f in self.stateful:
            f.fit(train_df)
        return self

    def transform_stateful(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 2 columns. Must be preceded by fit_stateful on the right rows."""
        return self._concat(df, self.stateful)
