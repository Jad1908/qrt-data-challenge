import numpy as np

import src.config as C
from src.data import load_train
from src.features import FeaturePipeline, available_features
from src.validation import PurgedTimeSeriesSplit

print(
    available_features()
)  # {'alloc_hit_rate': 2, 'ret_cross_section': 1, 'ret_summary': 0}

train = load_train()
pipe = FeaturePipeline(["ret_summary", "ret_cross_section", "alloc_hit_rate"])

# Eager (stateless) path on the full frame
stateless = pipe.transform_stateless(train)
print("stateless cols:", list(stateless.columns))
print(stateless.describe().T[["mean", "std", "min", "max"]].round(4))
assert stateless.notna().all().all(), "stateless features should have no NaN"

# Per-fold (stateful) path: fit on fold-0 train, transform val
cv = PurgedTimeSeriesSplit.from_config(C.CVConfig())
tr_idx, va_idx = next(cv.split(train))
tr, va = train.iloc[tr_idx], train.iloc[va_idx]

pipe.fit_stateful(tr)
hr_val = pipe.transform_stateful(va)
hr_col = hr_val.iloc[:, 0]
print(
    "\nalloc_hit_rate on val — mean:",
    round(float(hr_col.mean()), 4),
    "std:",
    round(float(hr_col.std()), 4),
    "prior:",
    round(pipe.stateful[0].prior_, 4),
)

# Leak test (unchanged logic, .values comparison is fine on the DataFrame)
va_corrupt = va.copy()
va_corrupt[C.TARGET_SIGN_COL] = 1 - va_corrupt[C.TARGET_SIGN_COL]
hr_val_corrupt = pipe.transform_stateful(va_corrupt)
assert np.allclose(hr_val.values, hr_val_corrupt.values), "LEAK!"
print("Leak test PASSED.")

# Correctness spot-check: a training row's hit-rate uses only strictly-earlier rows
f = pipe.stateful[0]
some_alloc = int(tr[C.ALLOC_ORDER_COL].iloc[0])
rows = tr[tr[C.ALLOC_ORDER_COL] == some_alloc].sort_values(C.TS_ORDER_COL)
print(
    f"\nalloc {some_alloc}: first row hit-rate should equal prior "
    f"(no history): {float(f.transform(rows.head(1)).iloc[0,0]):.4f} vs prior {f.prior_:.4f}"
)
