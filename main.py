import numpy as np
from src.data import load_train
from src.validation import PurgedTimeSeriesSplit
import src.config as C

train = load_train()
cv = PurgedTimeSeriesSplit(n_splits=5, embargo=C.N_LAGS, scheme="expanding")

# 1) Eyeball the fold layout
print(cv.describe(train).to_string(index=False))

# 2) Hard invariants, checked on actual indices
for fold, (tr, va) in enumerate(cv.split(train)):
    tr_dates = train.iloc[tr][C.TS_ORDER_COL]
    va_dates = train.iloc[va][C.TS_ORDER_COL]
    assert len(np.intersect1d(tr, va)) == 0, "row overlap!"
    # forward split: every training date strictly precedes every val date
    assert tr_dates.max() < va_dates.min(), "train not before val!"
    # the embargo actually opened a >= N_LAGS gap in date-positions
    gap = va_dates.min() - tr_dates.max()
    print(f"fold {fold}: train≤{tr_dates.max()}  val≥{va_dates.min()}  "
          f"value-gap={gap}  (rows {len(tr):,}/{len(va):,})")