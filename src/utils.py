import numpy as np
import yaml
from pathlib import Path


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return rmse, r2


def load_configs():
    configs_path = Path("./files/configs.yaml")

    with open(configs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)