import src.config as C
from src.data import load_test, load_train

train = load_train()
test = load_test()

print("train:", train.shape)  # ~ (527073, ~46)
print("test :", test.shape)  # ~ (31870, ~45)  -- same minus target cols

# Contract spot-checks
assert train[C.ID_COL].is_unique
assert train[C.TARGET_COL].notna().all()
assert C.TARGET_COL not in test.columns
assert train[C.TS_ORDER_COL].is_monotonic_increasing  # canonical sort held

print("dates:", train[C.TS_ORDER_COL].min(), "->", train[C.TS_ORDER_COL].max())
print("base rate P(target>0):", round(train[C.TARGET_SIGN_COL].mean(), 4))
print(
    train[
        [
            C.TS_COL,
            C.TS_ORDER_COL,
            C.ALLOCATION_COL,
            C.ALLOC_ORDER_COL,
            C.TARGET_COL,
            C.TARGET_SIGN_COL,
        ]
    ].head()
)
