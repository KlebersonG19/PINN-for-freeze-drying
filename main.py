import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.pinn.model import ParametricPINN
from src.pinn.training import Trainer
from src.pinn.evaluate import predict
from src.utils import load_configs


def main():
    # Citrus: seed 0; Strawberry: seed 33
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = load_configs()

    # Data
    DATASET_PATH = Path("./data/citrus_data.csv")
    df = pd.read_csv(DATASET_PATH) 
    X_data = torch.tensor(df[['tempo_min', 'espessura_m', 'pressao_pa']].values, dtype=torch.float32)
    y_data = torch.tensor(df[['mr_exp']].values, dtype=torch.float32)

    # Physics grid
    t_grid = np.linspace(0, cfg['model']['t_max'], cfg['physics_grid']['t_points'])
    L_grid = np.linspace(cfg['physics_grid']['L_min'], cfg['model']['L_max'], cfg['physics_grid']['L_points'])
    Pc_grid = np.linspace(cfg['physics_grid']['Pc_min'], cfg['model']['Pc_max'], cfg['physics_grid']['Pc_points'])

    T, L, P = np.meshgrid(t_grid, L_grid, Pc_grid)
    X_phys_np = np.hstack([T.flatten()[:, None], L.flatten()[:, None], P.flatten()[:, None]])
    X_phys = torch.tensor(X_phys_np, dtype=torch.float32)

    # Init model
    model = ParametricPINN(cfg['model']['t_max'], cfg['model']['L_max'], cfg['model']['Pc_max'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    trainer = Trainer(model, optimizer, epochs=cfg['training']['epochs'])
    trainer.train_model(X_data, y_data, X_phys)

    # Save model
    MODEL_NAME = "parametric_pinn_citrus_s0_new"
    save_path = Path(f"./files/models/{MODEL_NAME}.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # Evaluate
    print(f'\n{20*"="}PREDICTIONS{20*"="}')
    predict(model_path=save_path, ref_data="citrus")

if __name__ == '__main__':
    main()