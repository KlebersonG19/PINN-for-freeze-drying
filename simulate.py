import torch
import numpy as np
from pathlib import Path
import pandas as pd
from src.pinn.model import ParametricPINN
from src.visualizer.plot_graphs import plot_custom_scenario
from src.utils import load_configs


def predict_custom_scenario(model_path, L_mm, Pc_pa, target_mr=0.05, use_refs=None, save_plot=True):
    cfg = load_configs()

    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo de pesos '{model_path}' não encontrado.")

    model = ParametricPINN(cfg['model']['t_max'], cfg['model']['L_max'], cfg['model']['Pc_max'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Curvas de referencia
    curves_to_compare = []
    if use_refs:
        curves_to_compare = get_curves_to_compare(use_refs['data'], use_refs['targets'])

    # Ajustar para mudar grad view
    # t_continuous = np.linspace(0, 1920, 120)[:, None]
    t_continuous = np.linspace(0, 1920, 120)[:, None]
    L_m = L_mm / 1000.0
    
    X_custom = np.hstack([
        t_continuous, 
        np.full_like(t_continuous, L_m), 
        np.full_like(t_continuous, Pc_pa)
    ])
    
    with torch.no_grad():
        mr_pred_tensor, deff_pred_tensor = model(torch.tensor(X_custom, dtype=torch.float32))
        mr_pred = mr_pred_tensor.numpy()
        deff_estimado = deff_pred_tensor[0].item()

    idx_target = np.argmin(np.abs(mr_pred - target_mr))
    t_target = t_continuous[idx_target][0]
    
    print("=" * 60)
    print(f"- Espessura (L): {L_mm:.1f} mm ({L_m} m)")
    print(f"- Pressão da câmara (Pc):  {Pc_pa:.2f} Pa")
    print(f"- Difusividade Estimada (Deff): {deff_estimado:.2e} m²/s")
    print(f"- Tempo estimado para MR = {target_mr}: {t_target:.1f} min ({t_target/60:.2f} horas)")
    print(f"- Umidade restante (t={t_continuous[-1][0]:.0f} min): MR = {mr_pred[-1][0]:.4f}")
    print("=" * 60)

    plot_custom_scenario(t_continuous, mr_pred, L_mm, Pc_pa, t_target, target_mr, curves_to_compare, save_plot)


def get_curves_to_compare(data, targets):
    curves_to_compare = []

    if data == "citrus":
        df = pd.read_csv('data/citrus_data.csv')
    elif data == "strawberry":
        df = pd.read_csv('data/strawberry_data.csv')
    else:
        raise Exception(f"CSV não configurado para <{data}>.")
        
    for target in targets:
        L_ref = target['L']
        Pc_ref = target['Pc']
        
        df_filtered = df[(np.isclose(df['espessura_m'], L_ref)) & (np.isclose(df['pressao_pa'], Pc_ref))]
        
        if not df_filtered.empty:
            curves_to_compare.append({
                'L': L_ref,
                'Pc': Pc_ref,
                't': df_filtered['tempo_min'].values,
                'mr': df_filtered['mr_exp'].values
            })
        else:
            print(f"Curva de referência (L={L_ref}, Pc={Pc_ref}) não encontrada.")

    return curves_to_compare


if __name__ == '__main__':
    model_path = Path("files/models/parametric_pinn_citrus_s0_relu.pt")
    L_mm = 4.0
    Pc_pa = 1.4
    target_mr = 0.01

    refs_to_compare = {
        "data": "citrus",
        "targets": [
            {"L": 0.005, "Pc": 3.0},
            {"L": 0.007, "Pc": 3.0}
        ]
    }

    predict_custom_scenario(model_path, L_mm, Pc_pa, target_mr, use_refs=None, save_plot=True)