import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.pinn.model import ParametricPINN
from src.utils import calculate_metrics, load_configs
from src.visualizer.plot_graphs import plot_validation_grid


def predict(model_path: Path, ref_data="citrus", save_image=True):
    cfg = load_configs()

    if ref_data == "citrus":
        df = pd.read_csv('data/citrus_data.csv')
        espessuras_m = cfg[ref_data]['l_values']
        pressoes_pa = cfg[ref_data]['p_values']
    elif ref_data == "strawberry":
        df = pd.read_csv('data/strawberry_data.csv')
        espessuras_m = cfg[ref_data]['l_values']
        pressoes_pa = cfg[ref_data]['p_values']
    else:
        raise Exception(f"CSV não configurado para <{ref_data}>.")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo de pesos '{model_path}' não encontrado.")

    model = ParametricPINN(cfg['model']['t_max'], cfg['model']['L_max'], cfg['model']['Pc_max'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results_dict = {}

    for L in espessuras_m:
        for Pc in pressoes_pa:
            df_sub = df[(df['espessura_m'] == L) & (df['pressao_pa'] == Pc)]
            t_exp = df_sub['tempo_min'].values.astype(float)
            mr_exp = df_sub['mr_exp'].values

            # Tensors
            t_continuous = np.linspace(0, t_exp.max(), cfg[ref_data]['num_points'])[:, None]
            L_arr = np.full_like(t_continuous, L)
            Pc_arr = np.full_like(t_continuous, Pc)
            X_test_tensor = torch.tensor(np.hstack([t_continuous, L_arr, Pc_arr]), dtype=torch.float32)

            with torch.no_grad():
                mr_pred_tensor, deff_pred_tensor = model(X_test_tensor)
                mr_pred_continuous = mr_pred_tensor.numpy()
                deff_estimado = deff_pred_tensor[0].item()

                X_exp_np = np.hstack([t_exp[:, None], np.full_like(t_exp[:, None], L), np.full_like(t_exp[:, None], Pc)])
                mr_exp_tensor, _ = model(torch.tensor(X_exp_np, dtype=torch.float32))
                mr_pred_at_exp = mr_exp_tensor.numpy().flatten()

            # Metrics
            rmse, r2 = calculate_metrics(mr_exp, mr_pred_at_exp)
            print(f"L = {L*1000:.0f}mm | Pc = {Pc:.1f}Pa | RMSE: {rmse:.4f} | R²: {r2:.4f} | Deff: {deff_estimado:.2e} m²/s")

            results_dict[(L, Pc)] = {
                't_exp': t_exp,
                'mr_exp': mr_exp,
                't_continuous': t_continuous,
                'mr_pred': mr_pred_continuous,
                'rmse': rmse,
                'r2': r2
            }

    plot_validation_grid(espessuras_m, pressoes_pa, results_dict, save_image)

if __name__ == '__main__':
    direct_access_path = Path("files/models/parametric_pinn_citrus_s0_relu.pt")
    predict(model_path=direct_access_path, ref_data="citrus", save_image=True)


    # python -m src.pinn.evaluate