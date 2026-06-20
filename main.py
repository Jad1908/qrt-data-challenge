import numpy as np

import src.config as C
from src.data import load_train
from src.evaluate import evaluate_oof
from src.features import FeaturePipeline
from src.models import make_model, target_column_for
from src.validation import PurgedTimeSeriesSplit

train = load_train()
pipe = FeaturePipeline(["ret_summary", "ret_cross_section"])
feats = pipe.transform_stateless(train)
cv = PurgedTimeSeriesSplit.from_config(C.CVConfig())

task = "classify"
ycol = target_column_for(task)

# Collect OOF predictions across all five folds
oof_idx, oof_pred, oof_fold = [], [], []
for k, (tr_idx, va_idx) in enumerate(cv.split(train)):
    m = make_model("linear", task=task, seed=42)
    m.fit(feats.iloc[tr_idx], train.iloc[tr_idx][ycol].to_numpy())
    oof_idx.append(va_idx)
    oof_pred.append(m.predict_label(feats.iloc[va_idx]))
    oof_fold.append(np.full(len(va_idx), k))

oof_idx = np.concatenate(oof_idx)
oof_pred = np.concatenate(oof_pred)
oof_fold = np.concatenate(oof_fold)

report = evaluate_oof(train.iloc[oof_idx], oof_pred, oof_fold)
print(report.summary())
print("\nper-fold:\n", report.per_fold.to_string(index=False))
