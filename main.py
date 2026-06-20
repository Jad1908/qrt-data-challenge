from src.config import ExperimentConfig
from src.experiment import run_experiment
from src.submission import write_submission

cfg2 = ExperimentConfig(
    name="baseline_with_hitrate",
    features=["ret_summary", "ret_cross_section", "alloc_hit_rate"],
    model="linear",
)
result = run_experiment(cfg2)

print(result.report.summary())  # should reproduce ~0.5206, edge +0.0142
print(
    "\ntest predictions:",
    len(result.test_pred),
    "| pred-up rate:",
    round(result.test_pred.mean(), 4),
)

path = write_submission(result.test_row_id, result.test_pred, cfg2.name)
print("wrote:", path)
