from src.config import DATA_DIR, RET_COLS, ExperimentConfig

cfg = ExperimentConfig()
print(cfg.to_dict())  # should print the full nested spec
print(len(RET_COLS), RET_COLS[0], RET_COLS[-1])  # 20 RET_1 RET_20
print(DATA_DIR.exists())  # True
