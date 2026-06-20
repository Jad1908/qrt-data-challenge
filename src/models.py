"""Model adapters: a uniform interface that hides the classify/regress fork.

Every model exposes:
    fit(X, y) -> self
    predict_score(X) -> 1d float array   (P(up) if classify, predicted
    return if regress)
    to_label(score)  -> 1d int array     (threshold: 0.5 for proba,
    0.0 for return)
    predict_label(X) -> to_label(predict_score(X))

So the orchestrator never branches on task: it always does score -> label.
The ONLY task-dependent things it must know are (a) which target column to
feed, given by target_column_for(task), and (b) nothing else.

Preprocessing (impute + scale) is OWNED here and fit per fold inside fit(), so
fold-train statistics never leak. Features stay a pure feature layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C


def target_column_for(task: str) -> str:
    """Which y column the orchestrator should pass to fit(), given the task."""
    if task == "classify":
        return C.TARGET_SIGN_COL  # 0/1 label
    if task == "regress":
        return C.TARGET_COL  # continuous return
    raise ValueError(f"unknown task {task!r}")


# ── Base + registry ──────────────────────────────────────────────────────────
class BaseModel:
    name: str = "base"

    def __init__(self, task: str, params: dict | None = None, seed: int = 42):
        if task not in ("classify", "regress"):
            raise ValueError(f"unknown task {task!r}")
        self.task = task
        self.seed = seed
        self.params = dict(params or {})
        self.threshold = 0.5 if task == "classify" else 0.0
        self.estimator = self._build()

    def _build(self):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModel":
        self.estimator.fit(X, np.asarray(y))
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return the natural score: P(up) for classify, predicted value for regress."""
        if self.task == "classify":
            return self.estimator.predict_proba(X)[:, 1]
        return self.estimator.predict(X)

    def to_label(self, score: np.ndarray) -> np.ndarray:
        # strict '>' matches the challenge's sign(x)=1 iff x>0; ties -> 0 (bet against)
        return (np.asarray(score) > self.threshold).astype("int8")

    def predict_label(self, X: pd.DataFrame) -> np.ndarray:
        return self.to_label(self.predict_score(X))


_REGISTRY: dict[str, type[BaseModel]] = {}


def register(cls: type[BaseModel]) -> type[BaseModel]:
    if cls.name in _REGISTRY:
        raise ValueError(f"duplicate model name {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def make_model(
    name: str, task: str, params: dict | None = None, seed: int = 42
) -> BaseModel:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](task=task, params=params, seed=seed)


def available_models() -> list[str]:
    return sorted(_REGISTRY)


# ── Linear: LogisticRegression (classify) / Ridge (regress) ──────────────────
@register
class LinearModel(BaseModel):
    """Regularized linear baseline. Impute(median)+StandardScaler fit per fold.

    The preprocessing is wrapped in an sklearn Pipeline, so calling fit() fits the
    imputer and scaler on THIS fold's training rows only — leak-safe by design.
    """

    name = "linear"

    def _build(self) -> Pipeline:
        if self.task == "classify":
            defaults = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"}
            defaults.update(self.params)
            est = LogisticRegression(random_state=self.seed, **defaults)
        else:
            defaults = {"alpha": 1.0}
            defaults.update(self.params)
            est = Ridge(random_state=self.seed, **defaults)
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("est", est),
            ]
        )


# ── LightGBM: classifier / regressor ─────────────────────────────────────────
@register
class LGBMModel(BaseModel):
    """Gradient boosting. Handles NaN and raw scales natively -> no preprocessing.

    lightgbm is imported lazily so the rest of the codebase runs without it.
    """

    name = "lgbm"

    def _build(self):
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as e:  # pragma: no cover
            raise ImportError("lightgbm not installed: `pip install lightgbm`") from e

        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 200,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "verbosity": -1,
        }
        defaults.update(self.params)
        cls = LGBMClassifier if self.task == "classify" else LGBMRegressor
        return cls(random_state=self.seed, **defaults)
