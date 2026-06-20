import src.config as C
from src.data import load_train
from src.features import FeaturePipeline
from src.models import available_models, make_model, target_column_for
from src.validation import PurgedTimeSeriesSplit

print("models:", available_models())  # ['lgbm', 'linear']

train = load_train()
pipe = FeaturePipeline(
    ["ret_summary", "ret_cross_section"]
)  # stateless only, keeps test simple
feats = pipe.transform_stateless(train)

cv = PurgedTimeSeriesSplit.from_config(C.CVConfig())
tr_idx, va_idx = next(cv.split(train))
Xtr, Xva = feats.iloc[tr_idx], feats.iloc[va_idx]

for task in ("classify", "regress"):
    ycol = target_column_for(task)
    ytr = train.iloc[tr_idx][ycol].to_numpy()
    m = make_model("linear", task=task, seed=42).fit(Xtr, ytr)
    score = m.predict_score(Xva)
    label = m.to_label(score)
    print(f"\n[{task}] target={ycol}")
    print(f"  score range: [{score.min():.4f}, {score.max():.4f}]")
    print(f"  label mean (P predicted=1): {label.mean():.4f}")
    # quick accuracy against the true sign — should be near base rate at this stage
    truth = train.iloc[va_idx][C.TARGET_SIGN_COL].to_numpy()
    print(f"  fold-0 accuracy: {(label == truth).mean():.4f}")
