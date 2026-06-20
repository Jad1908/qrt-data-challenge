"""Orchestrator: config in -> (validated report, test predictions) out.

This is the ONLY module that knows the run order, and the ONLY place that enforces
the leak-safety contract operationally:

  * Tier 0/1 (stateless) features: computed ONCE on the full frame (safe).
  * Tier 2 (stateful) features: fit INSIDE the fold loop on TRAIN rows only,
    then applied to val; for submission, fit on ALL train, applied to test.
  * Model preprocessing (impute/scale) is fit per fold inside the model adapter.

So no validation/test row ever informs a feature or a scaler. run_experiment does
both jobs from one config: cross-validate (honest score) AND refit-on-everything
(test predictions), guaranteeing the scored model and the submitted model match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config as C
from src.data import load_test, load_train
from src.evaluate import EvalReport, evaluate_oof
from src.features import FeaturePipeline
from src.models import BaseModel, make_model, target_column_for
from src.validation import PurgedTimeSeriesSplit


@dataclass
class ExperimentResult:
    """Everything one run produces: the honest score + the deployable prediction."""

    config: "C.ExperimentConfig"
    report: EvalReport
    oof_pred: np.ndarray  # 0/1, aligned to oof_row_id
    oof_row_id: np.ndarray  # ROW_ID for each OOF prediction
    test_pred: np.ndarray  # 0/1, aligned to test_row_id
    test_row_id: np.ndarray
    test_score: np.ndarray  # raw scores (proba/return) for threshold work
    final_model: BaseModel  # the all-train model used for the submission


def _build_design(
    pipe: FeaturePipeline,
    stateless: pd.DataFrame,
    rows: pd.DataFrame,
    idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assemble the model design matrix for a set of rows.

    Stateless columns are sliced from the precomputed frame; stateful columns are
    transformed live (the caller must have fit them on the right training rows).
    """
    sl = stateless.iloc[idx] if idx is not None else stateless
    target_rows = rows.iloc[idx] if idx is not None else rows
    sf = pipe.transform_stateful(target_rows)
    if sf.shape[1] == 0:
        return sl
    return pd.concat([sl, sf.set_index(sl.index)], axis=1)


def _fit_on(
    cfg: "C.ExperimentConfig",
    pipe: FeaturePipeline,
    stateless: pd.DataFrame,
    train_rows: pd.DataFrame,
    train_idx: np.ndarray | None,
) -> tuple[BaseModel, pd.DataFrame]:
    """Fit stateful features + model on a set of training rows. Returns the model
    and the fitted design matrix. Used identically by CV folds and the final refit.
    """
    fit_rows = train_rows.iloc[train_idx] if train_idx is not None else train_rows
    pipe.fit_stateful(fit_rows)  # Tier-2 fit: TRAIN ROWS ONLY
    X = _build_design(pipe, stateless, train_rows, train_idx)
    ycol = target_column_for(cfg.task)
    y = fit_rows[ycol].to_numpy()
    model = make_model(cfg.model, task=cfg.task, params=cfg.model_params, seed=cfg.seed)
    model.fit(X, y)
    return model, X


def run_experiment(
    cfg: "C.ExperimentConfig",
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
) -> ExperimentResult:
    """Cross-validate AND produce a test prediction, all from one config."""
    if train is None:
        train = load_train()
    if test is None:
        test = load_test()

    pipe = FeaturePipeline(cfg.features)
    # Stateless features once on each frame (Tier 0/1 are safe on the full frame).
    stateless_train = pipe.transform_stateless(train)
    stateless_test = pipe.transform_stateless(test)
    cv = PurgedTimeSeriesSplit.from_config(cfg.cv)

    # ── Cross-validation: the honest score ───────────────────────────────────
    oof_idx, oof_pred, oof_fold = [], [], []
    for k, (tr_idx, va_idx) in enumerate(cv.split(train)):
        model, _ = _fit_on(cfg, pipe, stateless_train, train, tr_idx)
        Xva = _build_design(pipe, stateless_train, train, va_idx)
        oof_idx.append(va_idx)
        oof_pred.append(model.predict_label(Xva))
        oof_fold.append(np.full(len(va_idx), k))

    oof_idx = np.concatenate(oof_idx)
    oof_pred = np.concatenate(oof_pred)
    oof_fold = np.concatenate(oof_fold)
    report = evaluate_oof(train.iloc[oof_idx], oof_pred, oof_fold)

    # ── Final refit on ALL train -> test prediction ──────────────────────────
    final_model, _ = _fit_on(cfg, pipe, stateless_train, train, None)
    Xtest = _build_design(pipe, stateless_test, test, None)
    test_score = final_model.predict_score(Xtest)
    test_pred = final_model.to_label(test_score)

    return ExperimentResult(
        config=cfg,
        report=report,
        oof_pred=oof_pred,
        oof_row_id=train.iloc[oof_idx][C.ID_COL].to_numpy(),
        test_pred=test_pred,
        test_row_id=test[C.ID_COL].to_numpy(),
        test_score=test_score,
        final_model=final_model,
    )
