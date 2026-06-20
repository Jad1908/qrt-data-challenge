"""Evaluation + diagnostics.

A single accuracy number lies in a 0.51 regime: it hides whether the edge is real
(survives across dates/folds), how much is genuine skill vs. just leaning on the
majority class, and whether a delta is within noise. Every function here exists to
answer one of those three questions.

The accuracy label convention matches the challenge exactly:
    predicted 1 iff score > threshold ; truth 1 iff target > 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config as C


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def majority_baseline_accuracy(y_true: np.ndarray) -> tuple[float, int]:
    """Accuracy of always predicting the most common class, and that class.

    This is the number a model must beat to have done ANYTHING. On a tilted set,
    this is often already > 0.50 — which is exactly the 'free accuracy from
    imbalance' we want to separate out from real skill.
    """
    y = np.asarray(y_true)
    p_up = y.mean()
    return (max(p_up, 1 - p_up), int(p_up >= 0.5))


def accuracy_se(n: int, p: float = 0.5) -> float:
    """Standard error of an accuracy estimate over n rows. ~0.0017 at n=88k."""
    return float(np.sqrt(p * (1 - p) / max(n, 1)))


@dataclass
class EvalReport:
    """Everything you need to judge one experiment, in one printable object."""

    overall_acc: float
    baseline_acc: float
    edge: float  # overall_acc - baseline_acc  (skill over tilt)
    overall_se: float
    pred_up_rate: float  # fraction predicted 1 (reveals tilt)
    true_up_rate: float  # fraction truly 1
    per_fold: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_date: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        f = self.per_fold
        fold_str = "  ".join(f"{a:.4f}" for a in f["acc"]) if len(f) else "n/a"
        return (
            f"OOF accuracy : {self.overall_acc:.4f}  (±{self.overall_se:.4f})\n"
            f"  baseline   : {self.baseline_acc:.4f}  (always predict majority)\n"
            f"  edge       : {self.edge:+.4f}  "
            f"({'>' if self.edge > 2 * self.overall_se else '~'} "
            f"{'significant' if self.edge > 2 * self.overall_se else 'within noise'})\n"
            f"  pred up    : {self.pred_up_rate:.4f}   \
                true up: {self.true_up_rate:.4f}\n"
            f"  per-fold   : {fold_str}\n"
            f"  per-date   : mean={self.per_date['acc'].mean():.4f}  "
            f"std={self.per_date['acc'].std():.4f}  "
            f"%dates>0.5={self.per_date['beats_half'].mean():.2%}"
        )


def _per_date_table(df: pd.DataFrame, truth_col: str, pred_col: str) -> pd.DataFrame:
    """Accuracy and up-rates per date — the distribution behind the headline."""

    def agg(g: pd.DataFrame) -> pd.Series:
        acc = (g[truth_col] == g[pred_col]).mean()
        return pd.Series(
            {
                "n": len(g),
                "acc": acc,
                "true_up": g[truth_col].mean(),
                "pred_up": g[pred_col].mean(),
                "beats_half": float(acc > 0.5),
            }
        )

    return df.groupby(C.TS_ORDER_COL, sort=True).apply(agg, include_groups=False)


def evaluate_oof(
    frame: pd.DataFrame,
    oof_pred: np.ndarray,
    fold_id: np.ndarray,
) -> EvalReport:
    """Build the full report from out-of-fold predictions.

    Parameters
    ----------
    frame    : the rows that received an OOF prediction (carries truth + ts_order).
    oof_pred : 0/1 predictions aligned to `frame`'s row order.
    fold_id  : which fold produced each prediction (for per-fold accuracy).
    """
    work = frame[[C.TS_ORDER_COL, C.TARGET_SIGN_COL]].copy()
    work["pred"] = np.asarray(oof_pred).astype("int8")
    work["fold"] = np.asarray(fold_id)

    truth = work[C.TARGET_SIGN_COL].to_numpy()
    pred = work["pred"].to_numpy()

    overall = accuracy(truth, pred)
    base, _ = majority_baseline_accuracy(truth)
    se = accuracy_se(len(truth))

    per_fold = (
        work.groupby("fold")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "acc": accuracy(g[C.TARGET_SIGN_COL], g["pred"]),
                    "baseline": majority_baseline_accuracy(
                        g[C.TARGET_SIGN_COL].to_numpy()
                    )[0],
                    "pred_up": g["pred"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_fold["edge"] = per_fold["acc"] - per_fold["baseline"]

    per_date = _per_date_table(work, C.TARGET_SIGN_COL, "pred")

    return EvalReport(
        overall_acc=overall,
        baseline_acc=base,
        edge=overall - base,
        overall_se=se,
        pred_up_rate=float(pred.mean()),
        true_up_rate=float(truth.mean()),
        per_fold=per_fold,
        per_date=per_date,
    )
